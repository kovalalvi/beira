# eeg-fmri-project
This is clean and good structure verison of previous eeg fmri repo


Structure.
**utils** 
  - models_arch/
  - get_datasets.py - Functions for working with raw EEG and fMRI
  - inference.py - Scripts for inference of our model Many2one, many2many and many2many window-based.
  - preproc.py - Functions for working with raw EEG and fMRI 
  - torch_dataset.py 
  - train_utils.py - Function for training, loss_functions and train steps.



**notebook** 
  - CrossCorr_Wav2VecAE_combine.ipynb - It is best model which aggregate cross corr and wavelet features. 
  - Wav2Vec_AE_CWL_Loss_all.ipynb - It is notebook which I use as template for experiments. 


**Examples** 
- Features: Wavlets
- Model: 1D AutoEncoder. 
- Loss: Corr + MSE + Manifold-based
![ts_all_losses](https://user-images.githubusercontent.com/55140479/159255531-e3e53d2e-d195-4e37-a66f-6635c3b007c3.png)


- Model. 1D AutoEncoder. 
- Loss: MSE
![ts_combine_features](https://user-images.githubusercontent.com/55140479/159255803-222d12f9-b503-4cc8-9413-9fc7281e54f7.png)
