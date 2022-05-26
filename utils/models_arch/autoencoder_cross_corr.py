import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader, Subset
from pytorch_model_summary import summary
import numpy as np


# hp_autoencoder = dict(n_electrodes=30,
#                      n_freqs = 16,
#                      n_channels_out=6,
#                      corr_proj_size = 64,
#                      channels = [128, 128, 128, 64], 
#                      kernel_sizes=[5, 5, 3],
#                      strides=[8, 8, 4], 
#                      dilation=[1, 1, 1], 
#                      decoder_reduce=4, 
#                      window_cross_corr=512)

# model = CrossCorrAutoEncoder1D(**hp_autoencoder)

# x = torch.zeros(4, 30, 16, 1024)
# print('Input size: ', x.shape)
# print('Output size: ',model(x).shape)
# print('')

# print(summary(model, x, show_input=True))

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
                 stride=1, dilation=1, p_conv_drop=0.1):
        super(ConvBlock, self).__init__()
        
        
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
    
def extract_cross_corr(x, th=0.):
    """
    Calculate cross correllation between electrodes for each band 
    Return triu and reshape into one vector.
    
    Input:
        x.shape [batch, elec, n_freqs, time]
        th - threshold for removeing trash connectivity. 
    Returns: 
    x_cross_vec. shape [batch, n_freqs, elec*(elec-1)//2] 
    """
    batch, elec, n_freq, time = x.shape
        
    # cross corr features 
    x = x.transpose(1, 2)
    x = x.reshape(batch*n_freq, elec, -1)

    x_corrs = torch.stack([torch.corrcoef(x_) for x_ in x])
    x_corrs = torch.nan_to_num(x_corrs)
    x_corrs = x_corrs.reshape(batch, n_freq, elec, elec)

    x_corrs = torch.where(torch.abs(x_corrs)>th, x_corrs, torch.zeros_like(x_corrs))

    triu_idxs = torch.triu_indices(elec, elec, offset=1)
    x_corrs_vec =  x_corrs[..., triu_idxs[0], triu_idxs[1]]

    return x_corrs_vec  


def get_sliding_window_cross_corr(x, window_cross_corr, step = None, th = 0.3):
    """
    Calculate cross corr matrices with desired window w/o overlapping 
    step: stride in pathifying, so you can calculate cross corr with overlapping.
    
    
    Get x with shape (batch, elec, n_freq, time)
    
    
    Return:
        n_patches = time//window_cross_corr if step is None.
        x with shape ( batch, vector_size, n_patches ) 
    """
    
    
    if step is None: 
        step = window_cross_corr
    batch, elec, n_freq, time = x.shape
    
    x_patches = x.unfold(dimension=-1, 
                         size = window_cross_corr, 
                         step = step)
    num_patches = x_patches.shape[-2]

    x_corr_vec = []
    for patch_idx in range(num_patches):
        x_tmp = x_patches.select(-2, patch_idx)
        x_tmp = extract_cross_corr(x_tmp, th = th)
        x_corr_vec.append(x_tmp)
    x_corr_vec = torch.stack(x_corr_vec, axis = 1) # [batch, n_patches, n_freq, vector_proj]

    x_corr_vec = x_corr_vec.reshape(batch,num_patches, -1)
    x_corr_vec = x_corr_vec.transpose(1, 2)
    return x_corr_vec

     
    
class CrossCorrAutoEncoder1D(nn.Module):
    """
    This is implementation of AutoEncoder1D model for time serias regression
    
    decoder_reduce  -size of reducing parameter on decoder stage. We do not want use a lot of features here.
    
    corr_proj_size - 
    """
    def __init__(self,
                 window_cross_corr,
                 n_electrodes=30,
                 n_freqs = 16,
                 n_channels_out=21,
                 corr_proj_size = 128,
                 channels = [8, 16, 32, 32], 
                 kernel_sizes=[3, 3, 3],
                 strides=[4, 4, 4], 
                 dilation=[1, 1, 1], 
                 decoder_reduce=1,):
        
        super(CrossCorrAutoEncoder1D, self).__init__()
        

        self.n_electrodes = n_electrodes
        self.n_freqs = n_freqs
        self.n_inp_features = n_freqs*n_electrodes
        self.n_channels_out = n_channels_out
        self.window_cross_corr = window_cross_corr
        
        ## cross corr parameters. 
        cross_corr_vector_dim = int(n_freqs * n_electrodes*(n_electrodes-1)/2)
        
        self.project = nn.Sequential(nn.Conv1d(cross_corr_vector_dim, corr_proj_size, 1),
                                     nn.Dropout(p=0.25),
                                     nn.ReLU(),
                                     )
        
        # Encoder 
        self.model_depth = len(channels)-1
        channels = np.array(channels)
        
        self.spatial_reduce = ConvBlock(self.n_inp_features, channels[0], kernel_size=3)
        self.encoder = nn.Sequential(*[ConvBlock(channels[i], 
                                                 channels[i+1], 
                                                 kernel_sizes[i],
                                                 stride=strides[i], 
                                                 dilation=dilation[i], 
                                                 p_conv_drop=0.3) for i in range(self.model_depth)])
        ## Decoder 
        # Reduce number of channels ( do not touch last one. 
        channels[:-1] = channels[:-1]//decoder_reduce
        channels[-1] = channels[-1] + corr_proj_size
        print('Channels for decoder', channels)
        
        self.mapping = ConvBlock(channels[-1], channels[-1], 1)
        
        # channels
        self.decoder = nn.Sequential(*[UpConvBlock(scale=strides[i],
                                                   in_channels=channels[i+1],
                                                   out_channels=channels[i],
                                                   kernel_size=kernel_sizes[i], 
                                                   p_conv_drop=0.3) for i in range(self.model_depth-1, -1, -1)])
        
        
        self.conv1x1_one = nn.Conv1d(channels[0], 
                                     self.n_channels_out, 
                                     kernel_size=1,
                                     padding='same')

    def forward(self, x):
        """
        x should be divisible by window_cross_corr
        Also it window_cross_corr should be divisible by prod (strides).
        """
        batch, elec, n_freq, time = x.shape
        
        # cross corr features 
        x_corr_vec = get_sliding_window_cross_corr(x, self.window_cross_corr, 
                                                   step=self.window_cross_corr, 
                                                   th=0.3)
        
        x_proj = self.project(x_corr_vec) # [batch, corr_proj_size, num_pathes]
        
        # wavelet features
        x = x.reshape(batch, -1, time)
        x = self.spatial_reduce(x)
        
        x = self.encoder(x)
        
        # aggregate features.
        x_proj_rep = F.interpolate(x_proj, size=x.shape[-1],  mode='nearest' )
        x = torch.cat([x, x_proj_rep], dim=1)
        
        x = self.mapping(x)
        
        x = self.decoder(x)
        x = self.conv1x1_one(x)
        
        return x

    

###---------------------------------------------------------------------------------------------###
# MATRIX AGGREGATION 


def extract_cross_corr_matrices(x, th=0.):
    """
    Calculate cross correllation between electrodes for each band 
    Return triu and reshape into one vector.
    
    Input:
        x.shape [batch, elec, n_freqs, time]
        th - threshold for removeing trash connectivity. 
    Returns: 
    x_cross_vec. shape [batch, n_freqs, elec*(elec-1)//2] 
    """
    batch, elec, n_freq, time = x.shape
        
    # cross corr features 
    x = x.transpose(1, 2)
    x = x.reshape(batch*n_freq, elec, -1)

    x_corrs = torch.stack([torch.corrcoef(x_) for x_ in x])
    x_corrs = torch.nan_to_num(x_corrs)
    x_corrs = x_corrs.reshape(batch, n_freq, elec, elec)

    x_corrs = torch.where(torch.abs(x_corrs)>th, x_corrs, torch.zeros_like(x_corrs))

    return x_corrs  

def get_sliding_window_cross_corr_matrices(x, window_cross_corr, step = None, th = 0.3):
    """
    Calculate cross corr matrices with desired window w/o overlapping 
    step: stride in pathifying, so you can calculate cross corr with overlapping.
    
    
    Get x with shape (batch, elec, n_freq, time)
    
    
    Return:
        n_patches = time//window_cross_corr if step is None.
        x with shape ( batch, vector_size, n_patches ) 
    """
    
    
    if step is None: 
        step = window_cross_corr
    batch, elec, n_freq, time = x.shape
    
    x_patches = x.unfold(dimension=-1, 
                         size = window_cross_corr, 
                         step = step)
    num_patches = x_patches.shape[-2]

    x_corr_vec = []
    for patch_idx in range(num_patches):
        x_tmp = x_patches.select(-2, patch_idx)
        x_tmp = extract_cross_corr_matrices(x_tmp, th = th)
        x_corr_vec.append(x_tmp)
    x_corr_vec = torch.stack(x_corr_vec, axis = 1) # [batch, n_patches, n_freq, elec, elec]

    return x_corr_vec


class MatrixBlock(nn.Module):
    """
    Transform crosscorr matrices by maultiplying left and right
    for rediuccing dimension 
    It should find some patterns in connectivity with other electrodes.
    Input is [batch, N, N]
    Output is [batch, N_out, N_out]

    To do: 
        
    """
    def __init__(self, in_channels, out_channels):
        super(MatrixBlock, self).__init__()
        
        self.left = nn.Parameter(torch.randn(out_channels, in_channels)*0.1, requires_grad=True)
        self.right = nn.Parameter(torch.zeros(in_channels, out_channels)*0.1, requires_grad=True)

        
    def forward(self, x):
        """
        LXR
        n_in, n_in -> n_out, n_out
        """
        
        left_x = torch.matmul(self.left, x)
        left_x_right = torch.matmul(left_x, self.right)
        return left_x_right
    
class ProjectMatrix(nn.Module):
    """
    Transform all matrices 
    Input shape is ->  [ batch, num_pathes, n_freq, elec, elec ] 
    
    Create Matrix tansformation for each freq separately
    use the same parameters for bathc and numb patches 
    Apply 
    
    Transform into [batch, latent_size, num_pathes]
    latent_size = n_freq*n_out^2
        
    """
    def __init__(self, n_freq, n_electrodes, n_out, corr_proj_size):
        super(ProjectMatrix, self).__init__()
        
        # create model for each cross corr matrix
        self.models = nn.ModuleList([ MatrixBlock(n_electrodes, n_out) for i in range(n_freq)])
        
        output_size = n_freq*n_out*n_out
        self.project = nn.Conv1d(output_size, corr_proj_size, 1)
        self.act = nn.ReLU()
        
    def forward(self, x):
        """
        batch, num_pathes, n_freq, elec, elec -> batch, latent_size, num_pathes
        """
        batch, num_pathes, n_freq, elec, elec = x.shape
        
        
        x = x.reshape((batch*num_pathes, n_freq, elec, elec)) 
        
        preds = []
        for freq_idx in range(n_freq):
            matrix = torch.select(x, dim=1, index = freq_idx)
            res = self.models[freq_idx](matrix)
            preds.append(res)
            
        preds = torch.stack(preds, dim = 1)
        preds = preds.reshape(batch, num_pathes, -1)
        preds = preds.transpose(1, 2)
        
        latent_x = self.project(preds)
        latent_x = self.act(latent_x)
        

        return latent_x


class CrossCorrAutoEncoder1D_Matrix(nn.Module):
    """
    This is implementation of AutoEncoder1D model for time serias regression
    
    decoder_reduce  -size of reducing parameter on decoder stage. We do not want use a lot of features here.
    
    corr_proj_size - 
    In that case we use step by step matrix aggregation for cross corr matrices
    """
    def __init__(self,
                 window_cross_corr,
                 n_electrodes=30,
                 n_freqs = 16,
                 n_channels_out=21,
                 corr_proj_size = 128,
                 channels = [8, 16, 32, 32], 
                 kernel_sizes=[3, 3, 3],
                 strides=[4, 4, 4], 
                 dilation=[1, 1, 1], 
                 decoder_reduce=1,):
        
        super(CrossCorrAutoEncoder1D_Matrix, self).__init__()
        

        self.n_electrodes = n_electrodes
        self.n_freqs = n_freqs
        self.n_inp_features = n_freqs*n_electrodes
        self.n_channels_out = n_channels_out
        self.window_cross_corr = window_cross_corr
        
        ## cross corr parameters.
        
        self.project_matrix = ProjectMatrix(n_freq = n_freqs, n_electrodes=n_electrodes , 
                                     n_out=8, corr_proj_size=corr_proj_size)
        
        # Encoder 
        self.model_depth = len(channels)-1
        channels = np.array(channels)
        
        self.spatial_reduce = ConvBlock(self.n_inp_features, channels[0], kernel_size=3)
        self.encoder = nn.Sequential(*[ConvBlock(channels[i], 
                                                 channels[i+1], 
                                                 kernel_sizes[i],
                                                 stride=strides[i], 
                                                 dilation=dilation[i], 
                                                 p_conv_drop=0.3) for i in range(self.model_depth)])
        ## Decoder 
        # Reduce number of channels ( do not touch last one. 
        channels[:-1] = channels[:-1]//decoder_reduce
        channels[-1] = channels[-1] + corr_proj_size
        print('Channels for decoder', channels)
        
        self.mapping = ConvBlock(channels[-1], channels[-1], 1, p_conv_drop=0.3)
        
        # channels
        self.decoder = nn.Sequential(*[UpConvBlock(scale=strides[i],
                                                   in_channels=channels[i+1],
                                                   out_channels=channels[i],
                                                   kernel_size=kernel_sizes[i], 
                                                   p_conv_drop=0.1) for i in range(self.model_depth-1, -1, -1)])
        
        
        self.conv1x1_one = nn.Conv1d(channels[0], 
                                     self.n_channels_out, 
                                     kernel_size=1,
                                     padding='same')

    def forward(self, x):
        """
        x should be divisible by window_cross_corr
        Also it window_cross_corr should be divisible by prod (strides).
        """
        batch, elec, n_freq, time = x.shape
        
        # cross corr features 
        x_corr_matrix = get_sliding_window_cross_corr_matrices(x, self.window_cross_corr, 
                                                            step=self.window_cross_corr, 
                                                            th=0.3)
        
        x_proj = self.project_matrix(x_corr_matrix) # [batch, corr_proj_size, num_pathes]
        
        # wavelet features
        x = x.reshape(batch, -1, time)
        x = self.spatial_reduce(x)
        
        x = self.encoder(x)
        
        # aggregate features.
        x_proj_rep = F.interpolate(x_proj, size=x.shape[-1],  mode='nearest' )
        x = torch.cat([x, x_proj_rep], dim=1)
        
        x = self.mapping(x)
        
        x = self.decoder(x)
        x = self.conv1x1_one(x)
        
        return x

    
    
###------------------------------------------------------------------###
# Temporal aggregation 



class CrossCorr_Matrix(nn.Module):
    """
    This is implementation of AutoEncoder1D model for time serias regression
    
    decoder_reduce  -size of reducing parameter on decoder stage. We do not want use a lot of features here.
    
    corr_proj_size - 
    In that case we use step by step matrix aggregation for cross corr matrices
    """
    def __init__(self,
                 window_cross_corr,
                 n_electrodes=30,
                 n_freqs = 16,
                 n_channels_out=21,
                 corr_proj_size = 128,
                 channels = [8, 16, 32, 32], 
                 kernel_sizes=[3, 3, 3],
                 strides=[4, 4, 4], 
                 dilation=[1, 1, 1], 
                 decoder_reduce=1,):
        
        super(CrossCorr_Matrix, self).__init__()
        

        self.n_electrodes = n_electrodes
        self.n_freqs = n_freqs
        self.n_inp_features = n_freqs*n_electrodes
        self.n_channels_out = n_channels_out
        self.window_cross_corr = window_cross_corr
        
        ## cross corr parameters.
        
        self.project_matrix = ProjectMatrix(n_freq = n_freqs, n_electrodes=n_electrodes , 
                                            n_out=8, corr_proj_size=corr_proj_size)
        
        # Encoder 
        self.model_depth = len(channels)-1
        channels = np.array(channels)
        
        ## Decoder 
        # Reduce number of channels ( do not touch last one. 
        channels[:-1] = channels[:-1]//decoder_reduce
        channels[-1] = corr_proj_size
        print('Channels for decoder', channels)
        
        self.mapping = ConvBlock(channels[-1], channels[-1], 1, p_conv_drop=0.3)
        
        # channels
        self.decoder = nn.Sequential(*[UpConvBlock(scale=strides[i],
                                                   in_channels=channels[i+1],
                                                   out_channels=channels[i],
                                                   kernel_size=kernel_sizes[i], 
                                                   p_conv_drop=0.1) for i in range(self.model_depth-1, -1, -1)])
        
        
        self.conv1x1_one = nn.Conv1d(channels[0], 
                                     self.n_channels_out, 
                                     kernel_size=1,
                                     padding='same')

    def forward(self, x):
        """
        x should be divisible by window_cross_corr
        Also it window_cross_corr should be divisible by prod (strides).
        """
        batch, elec, n_freq, time = x.shape
        
        # cross corr features 
        x_corr_matrix = get_sliding_window_cross_corr_matrices(x, self.window_cross_corr, 
                                                            step=self.window_cross_corr, 
                                                            th=0.3)
        
        x_proj = self.project_matrix(x_corr_matrix) # [batch, corr_proj_size, num_pathes]
        
        x_proj_rep = F.interpolate(x_proj, scale_factor = 2, mode='nearest' )
        x = x_proj_rep
        
        x = self.mapping(x)
        
        x = self.decoder(x)
        x = self.conv1x1_one(x)
        
        return x
    
    
    
    


