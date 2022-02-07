import torch 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader, Subset
from pytorch_model_summary import summary
import numpy as np



class SepConv1D(nn.Module):
    def __init__(self, nin, kernels_per_layer, nout, kernel_size =3):
        super(SepConv1D, self).__init__()
        self.depthwise = nn.Conv1d(nin, nin * kernels_per_layer, 
                                   kernel_size=kernel_size, 
                                   padding='same', 
                                   groups=nin)
        self.pointwise = nn.Conv1d(nin * kernels_per_layer, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
    
    
class Block1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, kernels_per_layer=2, stride=1):
        super(Block1D, self).__init__()
        
        self.downsample = nn.MaxPool1d(kernel_size=stride, stride=stride)
        self.sep_conv = SepConv1D(in_channels, 
                                  kernels_per_layer, 
                                  out_channels, 
                                  kernel_size=kernel_size)
        self.bn = nn.BatchNorm1d(in_channels)

        self.conv_branch = nn.Sequential(
                            SepConv1D(out_channels, kernels_per_layer, out_channels),
                            nn.BatchNorm1d(out_channels),
                            nn.ReLU(),
                            SepConv1D(out_channels, kernels_per_layer, out_channels))
    def forward(self, x):
        x = self.downsample(x)
        x = self.bn(x)
        x = self.sep_conv(x)
        
        x_branch = self.conv_branch(x)
        
        x = F.relu(x_branch + x)
        return x
        
    
class UpsampleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, scale=2):
        super(UpsampleConvBlock, self).__init__()
        
        self.upsample = nn.Upsample(scale_factor=scale, mode='linear')
        self.conv_block = Block1D(in_channels, out_channels, kernel_size)
            
    def forward(self, x_small):
        
        x_upsample = self.upsample(x_small)
        x = self.conv_block(x_upsample)
        
        return x
        
        
        
class AutoEncoder1D(nn.Module):
    """
    This is implementaiotn of AutoEncoder1D model for time serias regression
    Try to adopt fully convolutional network for fMRI decoding 
    1. Denoise stage - Use res blocks for it 
    2. Encode information using 1D convolution with big stride(now use max pool)
    3. Decode information using cat pass from previous levels.
    4. Map to only 21 ROI
    
        model = AutoEncoder1D(n_res_block = 1, channels = [8, 16, 32, 32], 
                      kernel_sizes=[5, 5, 5, 5], strides=[4, 4, 4, 4], n_channels_out = 21, n_channels_inp=30)

        x = torch.zeros(4, 30, 4096)
        print(summary(model, x, show_input=False))

    
    Time lenght of input and output the same. 
    """
    def __init__(self,n_channels_inp=30,n_channels_out=21,
                 n_res_block=1,
                 channels = [8, 16, 32], 
                 kernel_sizes=[3, 3, 3],
                 strides=[4, 4, 4],):
        
        super(AutoEncoder1D, self).__init__()
        
        self.n_res_block = n_res_block
        self.model_depth = len(channels)-1

        self.n_electrodes=  30
        self.n_freqs = int(n_channels_inp/30)
        
        self.spatial_reduce_2d = nn.Conv2d(self.n_electrodes, 16, kernel_size=1)
        
        
        self.spatial_reduce = nn.Conv1d(16*self.n_freqs, channels[0], kernel_size=1)
        
        
        # create downsample blcoks in Sequentional manner.
        self.downsample_blocks = nn.ModuleList([Block1D(channels[i], 
                                                        channels[i+1], 
                                                        kernel_sizes[i],
                                                        stride=strides[i]) for i in range(self.model_depth)])
        
        # create upsample blcoks in Sequentional manner.
        channels_reverse = channels[::-1]
        self.upsamle_blocks = nn.ModuleList([UpsampleConvBlock(channels_reverse[i], 
                                                               channels_reverse[i+1], 
                                                               kernel_sizes[i], 
                                                               scale=strides[i]) for i in range(self.model_depth)])
        
        self.mapping = Block1D(channels[-1], channels[-1], kernel_sizes[-1], stride = 1)
        
        self.upsample_last =nn.Upsample(scale_factor=strides[0], mode='linear')
        
        self.conv1x1 = nn.Conv1d(channels[0], n_channels_out, kernel_size=1, padding='same')
        
        scale_one = 2**int(np.sum(np.log2(strides)))
        self.upsample_one_time = nn.Upsample(scale_factor=scale_one, mode='linear')
        self.conv1x1_one = nn.Conv1d(channels[-1], n_channels_out, kernel_size=3, padding='same')


    def forward(self, x):
        """
        1. Denoise 
        2. Encode  
        3. Mapping 
        4. Decode 
        """
        batch, elec_freq, time = x.shape
        
        x = x.reshape(batch, self.n_electrodes, -1, time)
        x = self.spatial_reduce_2d(x)
        x = x.reshape(batch, -1, time)
        x = self.spatial_reduce(x)
        
        # encode information
        for i in range(self.model_depth):
            x = self.downsample_blocks[i](x)            
        
        # make mapping 
        x = self.mapping(x)
        
        # decode information        
#         for i in range(len(self.upsamle_blocks)):
#             x = self.upsamle_blocks[i](x)

#         x = self.conv1x1(x)
        
        x = self.upsample_one_time(x)
        x = self.conv1x1_one(x)
  
        return x
hp_autoencoder = dict(
                        n_res_block = 0, channels = [64, 64, 32, 32], 
                        kernel_sizes=[15, 11, 7, 5],
                        strides=[8, 8, 4],
                        n_channels_out = 21,
                        n_channels_inp= 900)


model = AutoEncoder1D(**hp_autoencoder)
print(summary(model, torch.zeros(4, 900, WINDOW_SIZE), show_input=False))