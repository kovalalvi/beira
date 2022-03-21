import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader, Subset
from pytorch_model_summary import summary
import numpy as np




class ConvBlock(nn.Module):
    """
    Input is [batch, emb, time]
    simple conv block from wav2vec 2.0 
        - conv
        - layer norm by embedding axis
        - activation
    To do: 
        add res blocks.
    """
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, dilation=1, p_conv_drop=0.3):
        super(ConvBlock, self).__init__()
        
        # use it instead stride. 
        
        self.conv1d = nn.Conv1d(in_channels, out_channels, 
                                kernel_size=kernel_size, 
                                bias=False, 
                                padding='same')
        
        self.norm = nn.LayerNorm(out_channels)
        self.activation = nn.GELU()
        self.drop = nn.Dropout(p=p_conv_drop)
        
        self.downsample = nn.MaxPool1d(kernel_size=stride, stride=stride)


        
    def forward(self, x):
        """
        - conv 
        - norm 
        - activation
        - downsample 

        """
    
        x = self.conv1d(x)
        
        # norm by last axis.
        x = torch.transpose(x, -2, -1) 
        x = self.norm(x)
        x = torch.transpose(x, -2, -1) 
        
        x = self.activation(x)
        x = self.drop(x)
        
        x = self.downsample(x)

        
        return x

    
    
    
    
class UpConvBlock(nn.Module):
    def __init__(self, scale, **args):
        super(UpConvBlock, self).__init__()
        self.conv_block = ConvBlock(**args)
        self.upsample = nn.Upsample(scale_factor=scale, mode='linear', align_corners=False)

            
    def forward(self, x):
        
        x = self.conv_block(x)
        x = self.upsample(x)
        return x    
    
    


    
    
class AutoEncoder1D(nn.Module):
    """
    This is implementation of AutoEncoder1D model for time serias regression
    
    decoder_reduce  -size of reducing parameter on decoder stage. We do not want use a lot of features here.
    """
    def __init__(self,
                 n_electrodes=30,
                 n_freqs = 16,
                 n_channels_out=21,
                 channels = [8, 16, 32, 32], 
                 kernel_sizes=[3, 3, 3],
                 strides=[4, 4, 4], 
                 dilation=[1, 1, 1], 
                 decoder_reduce=1):
        
        super(AutoEncoder1D, self).__init__()
        

        self.n_electrodes = n_electrodes
        self.n_freqs = n_freqs
        self.n_inp_features = n_freqs*n_electrodes
        self.n_channels_out = n_channels_out
        
        self.model_depth = len(channels)-1
        self.spatial_reduce = ConvBlock(self.n_inp_features, channels[0], kernel_size=3)
        
        # create downsample blcoks in Sequentional manner.
        self.downsample_blocks = nn.ModuleList([ConvBlock(channels[i], 
                                                        channels[i+1], 
                                                        kernel_sizes[i],
                                                        stride=strides[i], 
                                                        dilation=dilation[i]) for i in range(self.model_depth)])
        
        # make the same but in another side w/o last conv.
        channels = [ch//decoder_reduce for ch in channels[:-1]] + channels[-1:]
        # channels
        self.upsample_blocks = nn.ModuleList([UpConvBlock(scale=strides[i],
                                                          in_channels=channels[i+1],
                                                          out_channels=channels[i],
                                                          kernel_size=kernel_sizes[i]) for i in range(self.model_depth-1, -1, -1)])
        
        
        self.conv1x1_one = nn.Conv1d(channels[0], self.n_channels_out, kernel_size=1, padding='same')

    def forward(self, x):
        """
        """
        batch, elec, n_freq, time = x.shape
        
        x = x.reshape(batch, -1, time)
        x = self.spatial_reduce(x)
        
        # encode information
        for i in range(self.model_depth):
            x = self.downsample_blocks[i](x)
            
            
        for i in range(self.model_depth):
            x = self.upsample_blocks[i](x)     
        
        x = self.conv1x1_one(x)
        
        return x

