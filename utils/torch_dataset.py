
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm


class CreateDataset_eeg_fmri(Dataset):
    """
    Important time is last axis. 
    x_tensor - wavelet features - [ch, freq, time] # 
    y_tensor - hand pose - [21, time]
    
    Return: 
        x_crop - [ch, freq, window_size]
        y_crop - [21, window_size]
    """

    def __init__(self, dataset, 
                 window_size=1024, 
                 random_sample=False, 
                 sample_per_epoch=None, 
                 to_many = False):

        self.x, self.y = dataset
        self.WINDOW_SIZE = window_size
        self.start_max = self.x.shape[-1] - window_size - 1 

        self.random_sample = random_sample
        self.sample_per_epoch = sample_per_epoch
        self.to_many = to_many
        
    def __len__(self):
        if self.random_sample: 
            return self.sample_per_epoch
        return self.start_max

    def __getitem__(self, idx):
        
        if self.random_sample: 
            idx  = np.random.randint(0, self.start_max)
        
        start , end = idx, idx+self.WINDOW_SIZE
        eeg = self.x[..., start:end]
        
        if self.to_many:
            fmri = self.y[..., start:end]
        else:
            fmri = self.y[..., end-1]
        return (eeg, fmri)
    
    def get_full_dataset(self,inp_stride=1, step=1):
        """
        step - step between starting points neighbour points 
        inp_stride - take input with some stride( downsample additional). 
        """
        x_list = []
        y_list = []
        for idx in tqdm(range(0, len(self), step)):
            data = self[idx]
            x_list.append(data[0][..., ::inp_stride])
            y_list.append(data[1])
        x_list = np.array(x_list)
        y_list = np.array(y_list)
        return x_list, y_list 
        
        return 