import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader, Subset
from pytorch_model_summary import summary
import numpy as np

# **Change on 28 November**
# - Remove bias in all convolution 
# - Use GELU instead ReLu
# - Change batch on instance  normalization 
#     - It compute statisctics croos all time points for some channels. -> channels have same distribution.
#     - On inference we apply statisctics from training.
#     - Might be issue when we inferecen for all timeserais lenght.- Add more filters per channel in SepConv 
# - dilation the same.

class SepConv1D(nn.Module):
    def __init__(self, nin, kernels_per_layer, nout, kernel_size=3, dilation=1):
        super(SepConv1D, self).__init__()
        self.depthwise = nn.Conv1d(nin, nin * kernels_per_layer, 
                                   kernel_size=kernel_size, 
                                   padding='same', 
                                   groups=nin, 
                                   dilation=dilation, 
                                   bias = False)
        self.pointwise = nn.Conv1d(nin * kernels_per_layer, nout, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
    
    
class Block1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, kernels_per_layer=2, stride=1, dilation=1):
        super(Block1D, self).__init__()
        
        self.downsample = nn.AvgPool1d(kernel_size=stride, stride=stride)
        
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True)
        self.activation = nn.GELU()
        self.conv_branch = nn.Sequential(
                            SepConv1D(out_channels, kernels_per_layer, out_channels, dilation=dilation),
                            nn.GELU(),
                            SepConv1D(out_channels, kernels_per_layer, out_channels, dilation=dilation))
        
        self.norm = nn.InstanceNorm1d(out_channels, affine=False) # no trainable
        
    def forward(self, x):
        
        x = self.conv1d(x)
        x = self.activation(x)

        x_branch = self.conv_branch(x)
        
        x = self.norm(x_branch + x)
        
        x = self.downsample(x)


        return x
        
    
class UpsampleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, scale=2):
        super(UpsampleConvBlock, self).__init__()
        
        self.upsample = nn.Upsample(scale_factor=scale, mode='linear')
        self.conv_block = Block1D(in_channels, out_channels, kernel_size)
            
    def forward(self, x):
        
        x = self.conv_block(x)
        x = self.upsample(x)
        
        return x
        
        
        
class AutoEncoder1D(nn.Module):
    """
    This is implementaiotn of AutoEncoder1D model for time serias regression
    Try to adopt fully convolutional network for fMRI decoding 
    1. Denoise stage - Use res blocks for it 
    2. Encode information using 1D convolution with big stride(now use max pool)
    3. Decode information using cat pass from previous levels.
    4. Map to only 21 ROI
    



    model = AutoEncoder1D(**hp_autoencoder)
    print(summary(model, torch.zeros(4, 30, 30 , 1024), show_input=False))

    
    Time lenght of input and output the same. 
    """
    def __init__(self,n_electrodes=30,
                 n_freqs = 16,
                 n_channels_out=21,
                 n_res_block=1,
                 channels = [8, 16, 32, 32], 
                 kernel_sizes=[3, 3, 3],
                 strides=[4, 4, 4]):
        
        super(AutoEncoder1D, self).__init__()
        
        self.n_res_block = n_res_block
        self.model_depth = len(channels)-1

        self.n_electrodes = n_electrodes
        self.n_freqs = n_freqs
        
        ## factorized features reducing
        # electrodes spatial reducing
        self.spatial_reduce_2d = nn.Conv2d(self.n_electrodes, self.n_electrodes//2, kernel_size=1)
        
        # freqs-electrodes  reducing to channels[0].
        self.spatial_reduce = nn.Conv1d(self.n_electrodes//2*self.n_freqs,
                                        channels[0], kernel_size=1)
        
        
        # create downsample blcoks in Sequentional manner.
        self.downsample_blocks = nn.ModuleList([Block1D(channels[i], 
                                                        channels[i+1], 
                                                        kernel_sizes[i],
                                                        stride=strides[i], 
                                                        dilation=1) for i in range(self.model_depth)])
        
        
        # mapping in latent space
        self.mapping = Block1D(channels[-1], channels[-1],
                               kernel_sizes[-1], 
                               stride=1,
                               dilation=1)
        
        self.conv1x1_one = nn.Conv1d(channels[-1], n_channels_out, kernel_size=1, padding='same')

        
        # upscale one time to. Inverse Interpolation. 
        scale_one = 2**int(np.sum(np.log2(strides)))
        self.upsample_one_time = nn.Upsample(scale_factor=scale_one, mode='linear')


    def forward(self, x):
        """
        1. Denoise 
        2. Encode  
        3. Mapping 
        4. Decode 
        """
        batch, elec, n_freq, time = x.shape
        
        # x = x.reshape(batch, self.n_electrodes, -1, time)
        x = self.spatial_reduce_2d(x)
        
        x = x.reshape(batch, -1, time)
        x = self.spatial_reduce(x)
        
        # encode information
        for i in range(self.model_depth):
            x = self.downsample_blocks[i](x)            
        
        # make mapping 
        x = self.mapping(x)
        x = self.conv1x1_one(x)
        
        # inverse interpolation
        x = self.upsample_one_time(x)
        
        return x
