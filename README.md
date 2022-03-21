# eeg-fmri-project
This is clean and good structure verison of previous eeg fmri repo


Structure.
utils 
  - models_arch/
  - get_datasets.py - Functions for working with raw EEG and fMRI
  - inference.py - Scripts for inference of our model Many2one, many2many and many2many window-based.
  - preproc.py - Functions for working with raw EEG and fMRI 
  - torch_dataset.py 
  - train_utils.py - Function for training, loss_functions and train steps.
notebook 
  - CrossCorr_Wav2VecAE_combine.ipynb - It is best model which aggregate cross corr and wavelet features. 
  - Wav2Vec_AE_CWL_Loss_all.ipynb - It is notebook which I use as template for experiments. 


Examples 
