# BEIRA: BOLD from EEG Interpretable Regression Autoencoder

[**fMRI from EEG is only Deep Learning away: the use of interpretable DL to unravel EEG-fMRI relationships**](https://ommer-lab.com/research/latent-diffusion-models/)<br/>
[Alexander Kovalev](https://github.com/kovalalvi)\,
[Ilya ](https://github.com/ablattmann)\*,
[Alex Ossadtchi](https://github.com/qp-qp)

#### [[papers with code]](https://prompt-to-prompt.github.io/ptp_files/Prompt-to-Prompt_preprint.pdf)

#### [[Paper]](https://prompt-to-prompt.github.io/ptp_files/Prompt-to-Prompt_preprint.pdf) [[Project page]](https://prompt-to-prompt.github.io/ptp_files/Prompt-to-Prompt_preprint.pdf)




This repo is the official implementation of **"fMRI from EEG is only Deep Learning away: the use of interpretable DL to unravel EEG-fMRI relationships"**.
Here we present an interpretable domain grounded solution to recover the activity of several subcortical regions from the multichannel EEG data and demonstrate **up to
0.6 correlation** between the actual subcortical blood oxygenation level dependent (sBOLD) signal and its EEG-derived twin.



## Approach
![beira_net_arch-1](https://user-images.githubusercontent.com/55140479/197827587-8053d18a-193c-4795-9f0c-0b8fbb3505fe.png)


Figure 1: **Left** BOLD from EEG Interpretable Regression Autoencoder. We apply our architecture to a block of raw EEG data. First, the multibranch interpretable feature extraction module estimates physiologically interpretable features that then get compressed by the encoding block to be next unpacked with the decoder into the delayed window of ROI sBOLD activity samples. **Right** Building blocks. We use one layer interpretable compact block and several layers for encoder
and decoder as outlined in the diagram


## Results
CWL EEG/fMRI Dataset. Concurrently recording of EEG and fMRI data - [dataset link](https://paperswithcode.com/dataset/cwl-eeg-fmri-data-set)


<!-- ![ts_best_plots-1](https://user-images.githubusercontent.com/55140479/197828892-6b4993a7-9baa-4462-87d6-516f85d93dad.png) -->
<img src="https://user-images.githubusercontent.com/55140479/197828892-6b4993a7-9baa-4462-87d6-516f85d93dad.png" alt="drawing" width="600"/>
Figure 2. Real and predicted sBOLD time series of different RoI for one patient from CWL dataset: pallidum, caudate, putamen, accumbens. Yellow is real, blue is prediction. x axis - Time seconds. y axis - sBOLD activity


## How to start
For starting training procees you should prepare data.

    ├── utils 
    │   ├── models_arch
    │       ├── move1.npy
    │   ├── get_datasets.py  - build datasets from raw EEG and fMRI data    
    │   ├── inference.py - scripts for inference of our models. (there are many2one, many2many and many2many window-based).
    │   ├── preproc.py - preprocessing functions for raw EEG and fMRI  
    │   ├── get_datasets.py  - build datasets from raw EEG and fMRI data 
    │   ├── train_utils.py - all functions for model training, loss_functions
    └── ...

    ├── notebooks 
    │   ├── train_model.ipynb
    │   ├── inference_model.ipynb    
    └── ...



## Citation

```
@article{hertz2022prompt,
  title={fMRI from EEG is only Deep Learning away: the use of interpretable DL to unravel EEG-fMRI relationships},
  author={Hertz, Amir and Mokady, Ron and Tenenbaum, Jay and Aberman, Kfir and Pritch, Yael and Cohen-Or, Daniel},
  journal={arXiv preprint arXiv:2208.01626},
  year={2022}
}
```
