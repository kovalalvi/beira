# BEIRA
### [Paper](https://prompt-to-prompt.github.io/ptp_files/Prompt-to-Prompt_preprint.pdf)


This repo is the official implementation of "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows". Breakthroug paper which allows t






## Structure.

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
  title={Prompt-to-Prompt Image Editing with Cross Attention Control},
  author={Hertz, Amir and Mokady, Ron and Tenenbaum, Jay and Aberman, Kfir and Pritch, Yael and Cohen-Or, Daniel},
  journal={arXiv preprint arXiv:2208.01626},
  year={2022}
}
```
