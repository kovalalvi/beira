import mne
import os 
import json 

import numpy as np
import pandas as pd

from scipy.interpolate import interp1d
from nilearn import datasets, image, masking
from nilearn.input_data import NiftiLabelsMasker

# scripts 
# - general scripts 
# - CWL functions
# - NODDI functions


# -------------------------------------#
### GENERAL FUNCTION 




def atlas_masker(regions):
    """
    Return masker and labels for RoI extraction
    """
    dataset_cort = datasets.fetch_atlas_harvard_oxford("%s-maxprob-thr25-2mm" % regions)
    atlas_filename = dataset_cort.maps
    labels = dataset_cort.labels
    masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True)
    return masker, labels


def atlas(fmri_voxels):
    """
    Transform 4d  NIFTi type to ROI using masker 
    Return RoI and name of labels ( with first Background )
    """
    masker, labels = atlas_masker('sub')
    time_series = masker.fit_transform(fmri_voxels)
    return time_series, labels





def interpolate_df_eeg_fmri(df_eeg, df_fmri):
    """
    Make interpolation of fMRI data based on two DataFrames time columns in ms.
    And return value of fmri on each timestep of eeg data. 
    Then crop by fmri lenght of recording. Estimate fps of eeg data and return it.
    
    -----
    Input
    All input dataframes should have ['time'] columns in ms. 
    df_eeg - n_points x n_channels + 1
        df_eeg - should have more fps.
    
    df_fmri - n_points x n_roi + 1
    
    ------
    Output 
    df_eeg_align,
    df_fmri_interp,
    int(original_fps) original.
    """

    fmri_interpolate_func = interp1d(df_fmri['time'], 
                                     df_fmri.to_numpy(),
                                     kind='cubic',
                                     axis=0)  # [time, n_regions]


    # get all index of eeg that correspond with fMRI  min< idxs < max
    filter_rule = (df_eeg['time'] > df_fmri['time'].iloc[0]) & (df_eeg['time'] < df_fmri['time'].iloc[-1]) # problem with last element of pandas serias  
    df_eeg_croped = df_eeg[filter_rule]
    

    # make interpolation and then make pandas dataframe
    fmri_interpolate = fmri_interpolate_func(df_eeg_croped['time'])
    df_fmri_interp = pd.DataFrame(fmri_interpolate, columns=df_fmri.columns)
    
    
    # get fps value of signals
    original_fps = 1 / (df_eeg_croped['time'].iloc[1] - df_eeg_croped['time'].iloc[0]) * 1000

    return df_eeg_croped, df_fmri_interp, int(original_fps)



#---------------------------------------------------#

### CWL FUNCTION 

def retrive_times_fmri_cwl(raw):
    """
    Extract information from annottation raw file about fmri frames.
    This information importatn for further interpo
    
    -----
    Input
    Raw is file from EEG set.
    Retrive fMRI time annotation. When occurs recordings in seconds
    It is useful for aligning EEG and fMRI data 
    
    Output: 
    times_fmri - np 
        array of times in ms .
    """
    
    times_fmri = []
    for annot in raw.annotations:
        if annot['description'] == "mri":
            times_fmri.append(annot['onset'])

    times_fmri = np.array(times_fmri)
    times_fmri = times_fmri * 1000  # seconds to milliseconds
    
    return times_fmri



def extract_oxford_roi_cwl(fmri_img, t_r,
                       standartize=False, 
                       remove_confounds=False, 
                       motion_params_path=None, 
                       oxford_atlas_name = 'sub-maxprob-thr25-2mm'):
    """
    This function extract Oxford subcortical RoI from fMRI voxels data.
    Also you can remove confounds.
    
    Input: 
        t_r - repetition time in seconds. 
    
    list_oxford_atlas= ['sub-prob-2mm', "cort-maxprob-thr25-2mm", "sub-maxprob-thr25-2mm"]
    
    Return :
        RoI - (time, n_ROI)
        labels - n_ROI 
    """
    
    # calculate mask of brain to extract global signal
    mean_img = image.mean_img(fmri_img)
    mask = masking.compute_epi_mask(mean_img)
    masker_global_signal = NiftiLabelsMasker(mask, 'global_signal',
                                             detrend = False,
                                             standardize=False, 
                                             t_r = t_r)

    ts_global_signal = masker_global_signal.fit_transform(fmri_img)
     
    
    # choose variation of oxford atlas.

    atlas_sub = datasets.fetch_atlas_harvard_oxford(oxford_atlas_name)
    
    
    # previous results were obtained w/o detrend and with high_var_conf.
    masker = NiftiLabelsMasker(atlas_sub.maps, atlas_sub.labels,
                                t_r = t_r,
                                detrend = True, 
                                standardize=standartize,
                                high_variance_confounds=False)
    
    

    if remove_confounds: 
        global_signal_confound = pd.DataFrame(ts_global_signal)
        
        # add information about motion if path exist
        # if motion_params_path is not None: 
        motion_confound = pd.read_csv(motion_params_path, sep = '  ', header=None)
        assert motion_confound.shape[0] == global_signal_confound.shape[0], '!!!_problem with motion !!!'
        
        confounds = pd.concat([motion_confound, global_signal_confound], axis=1)
    else:
        confounds = None      

    
    roi = masker.fit_transform(fmri_img, confounds=confounds)

    return roi, atlas_sub.labels 

def download_cwl_dataset(patient, path_to_dataset, 
                                       remove_confounds=False, 
                                       verbose=True):
    """ 
    Download EEG and fMRI data from Eyes OpenClossed datasets
    Returns two dataframes with electrodes and RoI from fmri
    Remove_confounds - decrease bad high correlation between ROI.
    
    ------
    Input: 
    
    patient : str
         example - 'trio1'
    path_to_dataset
         exampel - ../data/eyes_open_closed_dataset
    
    return 
    df_eeg, df_fmri, labels
    """
    
    
    eeg_path_set_file = os.path.join(path_to_dataset, f'{patient}/CWL_Data/eeg/in-scan/{patient}_mrcorrected_eoec_in-scan_hpump-off.set')
    fmri_path_epi_raw = os.path.join(path_to_dataset, f'{patient}/CWL_Data/mri/epi_normalized/rwa{patient}_eoec_in-scan_hpump-off.nii')
    motion_params_path = os.path.join(path_to_dataset, f'{patient}/CWL_Data/mri/epi_motionparams/rp_a{patient}_eoec_in-scan_hpump-off.txt')

    if verbose ==True:
        print("ALL path: ", eeg_path_set_file, fmri_path_epi_raw, motion_params_path)
    

    ## EEG preprocessing
    raw = mne.io.read_raw_eeglab(eeg_path_set_file)

    df_eeg = raw.to_data_frame()
    vector_exclude = ['EOG', 'ECG', 'CW1', 'CW2', 'CW3', 'CW4', 'CW5', 'CW6']
    df_eeg = df_eeg.drop(vector_exclude, axis=1)
    
    ## fMRI processing 
    # get info about fmri data. 
    
    times_fmri = retrive_times_fmri_cwl(raw)
    t_r = (times_fmri[1] - times_fmri[0])/1000
    
    
    
    ## fMRI processing
    # - smooth fmri 
    # - extract RoI (optional remove confounds)
    fmri_im = image.smooth_img(fmri_path_epi_raw, fwhm=3)
    
    fmri_roi, labels = extract_oxford_roi_cwl(fmri_img=fmri_im,
                                              t_r = t_r,
                                              motion_params_path=motion_params_path, 
                                              remove_confounds=remove_confounds)
    
    
    
    # Weak point. Cause theree is problem. In eeg set 143 points but in fmri 146
    labels_roi=labels[1:]
    labels_roi[2]='Left Lateral Ventricle'
    
    df_fmri = pd.DataFrame(fmri_roi, columns=labels_roi)
    df_fmri = df_fmri[:len(times_fmri)]
    df_fmri['time'] = times_fmri
    df_fmri = df_fmri.drop(columns = ['Left Cerebral White Matter', 'Left Cerebral Cortex ', 
                             'Right Cerebral White Matter', 'Right Cerebral Cortex '])
    
    

    if verbose:
        print('Dimension of our EEG data: ', df_eeg.shape)
        print('Dimension of our fMRi data: ', fmri_im.shape)
        print('Dimension of our fMRi Roi data: ', df_fmri.shape)
        
        print('fMRI info : ', t_r)
        print('RoI: ', df_fmri.columns.to_list())

    return df_eeg, df_fmri, df_fmri.drop(['time'], 1).columns.to_list()





#------------------------------------------------------------#
### NODDI FUNCTION 
def retrive_times_fmri_noddi(raw):
    """
    Extract information from annottation raw file about fmri frames.
    This information importatn for further interpo
    
    -----
    Input
    Raw is file from EEG set.
    Retrive fMRI time annotation. When occurs recordings in seconds
    It is useful for aligning EEG and fMRI data 
    
    """
    times_fmri = []
    for annot in raw.annotations:
        if annot['description'] == "Scanner/Scan Start":
            times_fmri.append(annot['onset'])

    times_fmri = np.array(times_fmri)
    times_fmri = times_fmri * 1000  # seconds to milliseconds
    
    return times_fmri



def extract_oxford_roi_noddi(fmri_img, t_r, 
                             standartize=False, 
                             remove_confounds=False,
                             motion_params_path=None,
                             oxford_atlas_name = 'sub-maxprob-thr25-2mm'):
    """
    This function extract Oxford subcortical RoI from fMRI voxels data.
    Also you can remove confounds.
    Confounds are file with time serias which we should remove from original data. 
    
    Input: 
        t_r - repetition time in seconds. 

    list_oxford_atlas = ['sub-prob-2mm', "cort-maxprob-thr25-2mm", "sub-maxprob-thr25-2mm"]
    confounds_list = ['global_signal', 'csf', "white_matter",
                      "trans_x","trans_y",'trans_z', 'rot_x', 'rot_y', 'rot_z']

    Return :
        RoI - (time, n_ROI)
        labels - n_ROI 
    """


    # choose variation of oxford atlas.

    atlas_sub = datasets.fetch_atlas_harvard_oxford(oxford_atlas_name)
    
    masker = NiftiLabelsMasker(atlas_sub.maps, atlas_sub.labels, 
                               t_r = t_r, 
                               detrend=True,
                               standardize=standartize, 
                               high_variance_confounds=False)
    
    if remove_confounds:
        # process tsv file with condounds after fmriprep
        # choose tha main confounds from approx 200.
        # confounds_list = ['global_signal', 'csf', "white_matter", 
        #                   "trans_x","trans_y",'trans_z',
        #                   'rot_x', 'rot_y', 'rot_z']


        
        
        confounds_list = [ 'global_signal',
                          "trans_x","trans_y",'trans_z',
                          'rot_x', 'rot_y', 'rot_z']
        confounds_all = pd.read_csv(motion_params_path,  sep='\t', header=0)
        confounds = confounds_all[confounds_list]
        
    else: 
        confounds = None

    roi = masker.fit_transform(fmri_img, confounds=confounds)

    return roi, atlas_sub.labels 


def download_bids_noddi_dataset(patient, path_to_dataset, remove_confounds=True,verbose=True):
    """ 
    Download EEG and fMRI data from Eyes OpenClossed datasets
    Returns two dataframes with electrodes and RoI from fmri
    Remove_confounds - decrease bad high correlation between ROI.
    
    Example: 
        patient = '32'
        path_to_dataset = '../data/NODDI_dataset/'
        df_eeg, df_fmri, labels = download_bids_noddi_dataset(patient, path_to_dataset, 
                                                                remove_confounds=True,verbose=True)
    
    return 
    df_eeg - pandas dataframe 
        columns [electrode_1, electrode_1,... , time]
            time - in ms.
    
    df_fmri,
        columns [roi_1, roi_2,... , time]
            time - in ms.
    
    labels_roi
        name of each roi in df_fmri. 

    """
    
    
    fmri_preproc_path = os.path.join(path_to_dataset, 
                                     f"derivatives/sub-{patient}/func/sub-{patient}_task-rest_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz")
    motion_params_path = os.path.join(path_to_dataset, 
                                      f"derivatives/sub-{patient}/func/sub-{patient}_task-rest_desc-confounds_timeseries.tsv")
    fmri_desc_path = os.path.join(path_to_dataset, 
                                  f"derivatives/sub-{patient}/func/sub-{patient}_task-rest_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.json")
    
    eeg_preproc_path = os.path.join(path_to_dataset, f'derivatives/sub-{patient}/eeg/sub-{patient}_task-rest_eeg.vhdr')
  
    if verbose ==True:
        print("ALL path: ", fmri_preproc_path, motion_params_path, fmri_desc_path, eeg_preproc_path)
    
  
    
    ## EEG preprocessing
    raw = mne.io.read_raw_brainvision(eeg_preproc_path, preload=True, verbose=False)      
    raw.set_channel_types({'ECG': 'ecg'})
    raw, betas = mne.preprocessing.regress_artifact(raw, picks_artifact='ecg', verbose=True)
    
    df_eeg = raw.to_data_frame(picks='eeg')
    
    ## fMRI processing 
    # get info about fmri data. 
    fmri_info = json.load(open(fmri_desc_path))
    t_r = float(fmri_info['RepetitionTime'])

    try: 
        shift_time_fmri = fmri_info['StartTime'] # seconds
        shift_time_fmri = shift_time_fmri*1000 # in ms. 
    except:
        print('Not found time corrected data')
        shift_time_fmri = 0
        
    times_fmri = retrive_times_fmri_noddi(raw)
    times_fmri = times_fmri + shift_time_fmri
    
    
    ## fMRI processing
    # - smooth fmri 
    # - extract RoI (optional remove confounds)
    
    fmri_img = image.smooth_img(fmri_preproc_path, fwhm=3)
    
    fmri_roi, labels = extract_oxford_roi_noddi(fmri_img=fmri_img, 
                                                t_r = t_r, 
                                                motion_params_path=motion_params_path, 
                                                remove_confounds=remove_confounds)
    
     
    # remove background and rename one roi.
    labels_roi=labels[1:]
    labels_roi[2]='Left Lateral Ventricle'
    
    df_fmri = pd.DataFrame(fmri_roi, columns=labels_roi)
    df_fmri = df_fmri[:len(times_fmri)]
    df_fmri['time'] = times_fmri
    
    
    df_fmri = df_fmri.drop(columns = ['Left Cerebral White Matter', 'Left Cerebral Cortex ', 
                             'Right Cerebral White Matter', 'Right Cerebral Cortex '])
       
    
    if verbose:
        
        print('Dimension of our EEG data: ', df_eeg.shape)
        print('Dimension of our fMRi data: ', fmri_img.shape)
        print('Dimension of our fMRi Roi data: ', df_fmri.shape)
        
        print('fMRI info:', fmri_info)
        print('RoI: ', df_fmri.columns.to_list())
    
    return raw, df_eeg, df_fmri, df_fmri.drop(columns = ['time']).columns.to_list()

