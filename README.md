## BEIRA: BOLD from EEG Interpretable Regression Autoencoder
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/fmri-from-eeg-is-only-deep-learning-away-the/eeg-decoding-on-cwl-eeg-fmri-dataset)](https://paperswithcode.com/sota/eeg-decoding-on-cwl-eeg-fmri-dataset?p=fmri-from-eeg-is-only-deep-learning-away-the)

> #### [**fMRI from EEG is only Deep Learning away: the use of interpretable DL to unravel EEG-fMRI relationships**](https://arxiv.org/abs/2211.02024)<br/>
> [Alexander Kovalev](https://github.com/kovalalvi)\,
> [Ilya Mikheev]()\*,
> [Alex Ossadtchi]()
> #### [[Paper]](https://arxiv.org/pdf/2211.02024.pdf) [[Project page]]()
> **BEIRA** -  domain grounded interpretable convolutional model for predicting subcortical BOLD singals from EEG activity.


## Abstract 

The access to activity of subcortical structures offers unique opportunity for building intention dependent brain-computer interfaces, renders abundant options for exploring a broad range of cognitive phenomena in the realm of affective neuroscience including complex decision making processes and the eternal free-will dilemma and facilitates diagnostics of a range of neurological deceases. So far this was possible only using bulky, expensive and immobile fMRI equipment. Here we present an interpretable domain grounded solution to recover the activity of several subcortical regions from the multichannel EEG data and demonstrate up to 60% correlation between the actual subcortical blood oxygenation level dependent sBOLD signal and its EEG-derived twin. Then, using the novel and theoretically justified weight interpretation methodology we recover individual spatial and time-frequency patterns of scalp EEG predictive of the hemodynamic signal in the subcortical nuclei. The described results not only pave the road towards wearable subcortical activity scanners but also showcase an automatic knowledge discovery process facilitated by deep learning technology in combination with an interpretable domain constrained architecture and the appropriate downstream task.

## Approach
![beira_net_arch-1](https://user-images.githubusercontent.com/55140479/197827587-8053d18a-193c-4795-9f0c-0b8fbb3505fe.png)


We apply our architecture to a block of raw EEG data. First, the multibranch interpretable feature extraction module estimates physiologically interpretable features that then get compressed by the encoding block to be next unpacked with the decoder into the delayed window of ROI sBOLD activity samples. We use one layer interpretable compact block and several layers for encoder and decoder as outlined in the diagram.


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
@article{beira2022,
  title={fMRI from EEG is only Deep Learning away: the use of interpretable DL to unravel EEG-fMRI relationships},
  author={Kovalev, Alexander and Mikheev, Ilia and Ossadtchi, Alexei},
  journal={arXiv preprint arXiv:2211.02024},
  year={2022}
}
```
