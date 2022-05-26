import mne 
import numpy as np 
import pandas as pd 


def downsample_ts(data, factor=1, axis = -1): 
    if factor==1:
        return data
    data_ds = mne.filter.resample(data, 
                                 up=1, 
                                 down=factor, 
                                 axis=-1)
    return data_ds
def downsample_dataset(dataset, factor=1, axis =-1):
    x, y = dataset
    x_ds = downsample_ts(x, factor, axis)
    y_ds = downsample_ts(y, factor, axis)
    return (x_ds,y_ds) 



def divide_train_test(eeg, fmri, fps, test_sec):
    """
    Input should be numpy array with shapes: (channels, time,  ) 
    Divide on train and test data and convert into numpy.
    
    Return datasets with shapes (channels, time). 
    """
    test_size =  int(test_sec * fps) 
    
    eeg_train = eeg[..., :-test_size]
    eeg_test = eeg[..., -test_size:]
    
    fmri_train = fmri[..., :-test_size]
    fmri_test = fmri[..., -test_size:]
    
    print(f"Size of train dataset: eeg { eeg_train.shape} | fmri {fmri_train.shape}") 
    print(f"Size of test dataset: eeg { eeg_test.shape} | fmri {fmri_test.shape}") 
    
    return (eeg_train, fmri_train) , (eeg_test, fmri_test)

def filter_powerline_noise(data, sf, f=50, **filter_params):
    """
    Notch filter
    """
    harmonic_freqs = np.arange(f, sf//2, f)
    filtered_data = mne.filter.notch_filter(data, sf, harmonic_freqs, verbose=True)
    return filtered_data


def compute_wavelet(data, sf, freqs):
    """
    Example of freqs creation
        freqs = np.logspace(np.log10(10), np.log10(300), 20)
    """
    n_chan = data.shape[0]
    wavelet_power = mne.time_frequency.tfr_array_morlet(data.reshape((1, n_chan, -1)),
                                                        sfreq=sf,
                                                        freqs=freqs,
                                                        output='power')
    return wavelet_power[0], freqs


def normalize_data(data, means_stds=None):
    """
    data - normalize by second.
    means_std - (means, stds ) 

    """
    if means_stds is None:
        means = np.mean(data, axis=-1, keepdims=True)
        stds = np.std(data, axis=-1, keepdims=True)
        means_stds = (means, stds)

    
    transform_data = (data - means_stds[0]) / means_stds[1]
    
    return transform_data, means_stds

def low_level_preproc_eeg(eeg, fps):
    """
    Apply filtering and common rereference.
    eeg should be size ( n_electrodes, time)
    fps - Hz 
    """
    eeg_filter = filter_powerline_noise(eeg, sf=fps, verbose=True)
    eeg_filter = mne.filter.filter_data(eeg_filter, sfreq=fps, 
                                        l_freq=1, h_freq=100, 
                                        verbose=True)
    
    common_average = np.mean(eeg_filter, axis=0, keepdims=True)
    eeg_filter = eeg_filter - common_average
    
    return eeg_filter

    
def preproc_dataset(dataset, fps, freqs, crop_size=None,
                    inp_means_stds=None, out_means_stds=None):
    """
    Common reference rereferencing.
    Filter.
    Extract features.
    dataset -  ( eeg (64, 146460) ,  fmri (21, 146460) )
    
    means_std - (means, stds ) 
    """
    eeg_arrays, fmri_arrays = dataset
    
    
    ## eeg preproc
    

    # filter eeg data.
    eeg_filter = filter_powerline_noise(eeg_arrays, sf=fps, verbose=False)
    eeg_filter = mne.filter.filter_data(eeg_filter, sfreq=fps, 
                                        l_freq=1, h_freq=100, 
                                        verbose=False)
    
    # subtract common average.
    common_average = np.mean(eeg_filter, axis=0, keepdims=True)
    eeg_filter = eeg_filter - common_average
    

    # normalize data 
    # eeg_filter, inp_means_stds = normalize_data(eeg_filter, inp_means_stds)
    fmri_arrays, out_means_stds = normalize_data(fmri_arrays, out_means_stds)

    
    # extract time freq representation.
    # eeg_wavelet_features, freqs = compute_wavelet(eeg_filter, sf=fps, freqs=freqs)
    eeg_wavelet_features = eeg_filter
    
    # remove bound.wavelent artifacts. 
    
    print(f"Size of eeg features {eeg_wavelet_features.shape} | fmri {fmri_arrays.shape}")
    return (eeg_wavelet_features, fmri_arrays), freqs, inp_means_stds, out_means_stds





def bold_time_delay_align(dataset, fps, bold_delay=0):
    """
    eeg and fmri in dataset should be time last.  
    bold_delay - temporal delay in seconds. 
    
    """
    shift = int(bold_delay*fps)
    eeg, fmri = dataset
    
    size = eeg.shape[-1]
    eeg = eeg[..., :size-shift]
    fmri = fmri[..., shift:]
    return (eeg, fmri)
    