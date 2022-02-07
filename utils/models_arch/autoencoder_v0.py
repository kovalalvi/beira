import torch 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader, Subset
from pytorch_model_summary import summary
import numpy as np







class ResBlock1D(nn.Module):
    def __init__(self, in_channels, kernel_size=9):
        super(ResBlock1D, self).__init__()
        
        self.conv_branch = nn.Sequential(
                            nn.Conv1d(in_channels, in_channels, groups=in_channels,  kernel_size=kernel_size, padding='same'),
                            nn.ReLU(),
                            nn.Conv1d(in_channels, in_channels,groups=in_channels, kernel_size=kernel_size, padding='same')
                            )
    def forward(self, x):
        x_branch = self.conv_branch(x)
        x = F.relu(x_branch + x)
        return x
        
class ConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ConvBlock1D, self).__init__() 
        
        # if stride is one no max pool
        self.conv_block = nn.Sequential(
                            nn.MaxPool1d(kernel_size=stride, stride=stride, dilation=1),
                            nn.Conv1d(in_channels, out_channels, 
                                      kernel_size=kernel_size, padding='same'),
                            nn.BatchNorm1d(out_channels),
                            nn.ReLU()
                            )           
    def forward(self, x):      
        return self.conv_block(x)
        
        
class UpsampleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, scale=2):
        super(UpsampleConvBlock, self).__init__()
        
        self.upsample = nn.Upsample(scale_factor=scale, mode='linear')
        self.conv_block = ConvBlock1D(in_channels, out_channels, kernel_size)
            
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

        
        self.resblocks = nn.ModuleList([ResBlock1D(n_channels_inp, kernel_size=9) for _ in range(n_res_block)])
        
        self.spatial_reduce = nn.Conv1d(n_channels_inp, channels[0], kernel_size=1, padding='same')
        
        
        # create downsample blcoks in Sequentional manner.
        self.downsample_blocks = nn.ModuleList([ConvBlock1D(channels[i], channels[i+1], 
                                                kernel_sizes[i], strides[i]) for i in range(self.model_depth)])
        
        # create upsample blcoks in Sequentional manner.
        
        channels_reverse = channels[::-1]
        self.upsamle_blocks = nn.ModuleList([UpsampleConvBlock(channels_reverse[i], channels_reverse[i+1], 
                                                               kernel_sizes[i], 
                                                               strides[i]) for i in range(self.model_depth)])
        
        self.mapping = ConvBlock1D(channels[-1], channels[-1], kernel_sizes[-1], stride = 1)
        
        self.upsample_last =nn.Upsample(scale_factor=strides[0], mode='linear')
        
        
        self.conv1x1 = nn.Conv1d(channels[0], n_channels_out, kernel_size=1, padding='same')

        
        
#         self.conv1x1 = nn.Sequential(
#                         nn.Conv1d(channels[1], channels[0], kernel_size=1, padding='same'),
#                         nn.ReLU(), 
#                         nn.Conv1d(channels[0], n_channels_out, kernel_size=1, padding='same')
#         )
                        

    def forward(self, x):
        """
        1. Denoise 
        2. Encode  
        3. Mapping 
        4. Decode 
        """
        # denoise stage n times make denoise
#         for i in range(self.n_res_block):
#             x=self.resblocks[i](x)
            
        x = self.spatial_reduce(x)
        
        # encode information
        for i in range(self.model_depth):
            x = self.downsample_blocks[i](x)            
        
        # make mapping 
        x = self.mapping(x)
        
        # decode information        
        for i in range(len(self.upsamle_blocks)):
            x = self.upsamle_blocks[i](x)

#         x = self.upsample_last(x)
        x = self.conv1x1(x)
  
        return x
    
        
        
                    
                
        
        