import torch 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader, Subset
from pytorch_model_summary import summary
import numpy as np

class VectorQuantizer1D(nn.Module):
    """
      num_embeddings - K number szie of codebook 
      embedding_dim - number of channels before codebook generation.
      """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer1D, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCT -> BTC
        
        batch, channels, times = inputs.shape
        
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        # just copy of gradients.
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BTC -> BCT
        return loss, quantized.permute(0, 2, 1).contiguous(), perplexity, encodings

      
    
class SepConv1D(nn.Module):
    def __init__(self, nin, kernels_per_layer, nout, kernel_size=3, dilation=1):
        super(SepConv1D, self).__init__()
        self.depthwise = nn.Conv1d(nin, nin * kernels_per_layer, 
                                   kernel_size=kernel_size, 
                                   padding='same', 
                                   groups=nin, 
                                   dilation=dilation, 
                                   bias = False)
        self.pointwise = nn.Conv1d(nin * kernels_per_layer, nout, kernel_size=1, bias=True)

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
        
        self.norm = nn.InstanceNorm1d(out_channels, affine=True) # trainable
        
    def forward(self, x):
        x = self.downsample(x)
        
        x = self.conv1d(x)
        x = self.activation(x)

        x_branch = self.conv_branch(x)
        
        x = self.norm(x_branch + x)

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
        
        
        
class VQ_AutoEncoder1D(nn.Module):
    """
    
    How to use this network.
    hp_autoencoder = dict(
                        channels = [64, 64, 64, 64], 
                        kernel_sizes=[7, 5, 3],
                        strides=[4, 4, 4],
                        n_channels_out = 21,
                        n_electrodes= 30,
                        n_freqs = 16, 
                        codebook_size=16)

    WINDOW_SIZE = 4096
    model = VQ_AutoEncoder1D(**hp_autoencoder)
    x = torch.zeros(4, 30, 16, WINDOW_SIZE)
    model(torch.zeros(4, 30, 16, WINDOW_SIZE))
    print(summary(model, torch.zeros(2, 30, 16, WINDOW_SIZE), show_input=False))
    """
    def __init__(self,n_electrodes=30,
                 n_freqs = 16,
                 n_channels_out=21,
                 channels = [8, 16, 32, 32], 
                 kernel_sizes=[3, 3, 3],
                 strides=[4, 4, 4], 
                 codebook_size=64):
        
        super(VQ_AutoEncoder1D, self).__init__()
        
        self.codebook_size = codebook_size
        self.model_depth = len(channels)-1

        self.n_electrodes = n_electrodes
        self.n_freqs = n_freqs
        channels_reverse = channels[::-1]
        
        # freqs-electrodes  reducing to channels[0].
        self.spatial_reduce = nn.Conv1d(self.n_electrodes*self.n_freqs, channels[0], kernel_size=1)
        
        
        # create downsample blcoks in Sequentional manner.
        self.downsample_blocks = nn.ModuleList([Block1D(channels[i],
                                                            channels[i+1], 
                                                            kernel_sizes[i],
                                                            stride=strides[i]) for i in range(self.model_depth)])
        
        
        self.vq_layer = VectorQuantizer1D(num_embeddings=self.codebook_size,
                                          embedding_dim=channels[-1],
                                          commitment_cost=0.25)
        
        
        self.upsample_blocks = nn.ModuleList([UpsampleConvBlock(channels_reverse[i], 
                                                       channels_reverse[i+1], 
                                                       kernel_sizes[i], 
                                                       scale=strides[i]) for i in range(self.model_depth)])
        
        

        self.conv1x1_one = nn.Conv1d(channels_reverse[-1], n_channels_out, kernel_size=1, padding='same')


    def forward(self, x):
        """
        1. Encode  
        2. Mapping 
        3. Decode 
        
        """
        batch, elec, n_freq, time = x.shape
                
        x = x.reshape(batch, -1, time)
        x = self.spatial_reduce(x)
        
        for conv in self.downsample_blocks:
            x = conv(x)   
        
        loss, x_quant, perp, encodings  = self.vq_layer(x)
                
        for conv in self.upsample_blocks:
            x_quant = conv(x_quant)   
            
        x_recon = self.conv1x1_one(x_quant)
        
        if self.training:
            return x_recon, loss, perp
        else:
            return x_recon
        
        
        
