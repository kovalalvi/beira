
import numpy as np 
import torch

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader, Subset

from pytorch_model_summary import summary



## EXAMPLE  

# n_electrodes = 64
# n_features = 16
# window_size = 4096
# n_time_points = 32 
# stride = window_size//n_time_points

# eeg = torch.ones([2, n_electrodes, n_features, window_size])
# print(eeg.shape)


# config_feature_extractor = dict(n_inp_features=n_features,
#                                  channels = [32, 32, 32, 32, 32], 
#                                  kernel_sizes=[9, 5, 3, 3, 3],
#                                  strides=[4, 4, 2, 2, 2], 
#                                  dilations = [1, 1, 1, 1, 1])


# config_vanilla_transformer = dict(n_electrodes = n_electrodes,
#                                  sequence_length = n_time_points,
#                                  embed_dim=n_features,
#                                  n_roi=21,
#                                  num_heads=4,
#                                  mlp_ratio=4,
#                                  num_layers=2,
#                                  attn_dropout=0.1,                
#                                  mlp_dropout=0.5)

# config_factorized_transformer = dict(n_electrodes = n_electrodes,
#                                  sequence_length = n_time_points,
#                                  embed_dim=32,
#                                  n_roi=21,
#                                  num_heads=4,
#                                  mlp_ratio=4,
#                                  num_layers_spatial=2,
#                                  num_layers_temporal=2,
#                                  attn_dropout=0.1,                
#                                  mlp_dropout=0.5)

# conv_wav2net = wav2vec_conv(**config_feature_extractor)
# model_wav2net = Wav2Vec_aggregate(conv_wav2net)



# vit = Vanilla_transformer(**config_vanilla_transformer)
# factorized_vit = Factorized_transformer(**config_factorized_transformer)


# model_v1 = Super_model(feature_extractor = get_strided_func(128), 
#                        transformer = vit)


# model_v2 = Super_model(feature_extractor = model_wav2net, 
#                        transformer = factorized_vit)



# y_hat = model_v1(eeg) 
# y_hat_2 = model_v2(eeg) 

# print('Input size', eeg.shape)
# print('Prediction size ', y_hat.shape)
# print('Prediction size 2', y_hat_2.shape)


# # print('Output size complex', eeg_complex_features.shape)
# # print('Output size simple ', eeg_simple_features.shape)



# # y_hat = vit(eeg_complex_features)
# # y_hat_2 = factorizes_vit(eeg_complex_features)






# print(summary(model_v1, eeg, show_input=False))
# print(summary(model_v2, eeg, show_input=False))
# # print(summary(factorizes_vit, eeg_complex_features, show_input=False))
















class Conv_block(nn.Module):
    """
    Input is [batch, emb, time]
    simple conv block from wav2vec 2.0 
        - conv
        - layer norm by embedding axis
        - activation
    To do: 
        add res blocks.
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, dilation=1):
        super(Conv_block, self).__init__()
        
        # use it instead stride. 
        self.downsample = nn.AvgPool1d(kernel_size=stride, stride=stride)
        
        self.conv1d = nn.Conv1d(in_channels, out_channels, 
                                kernel_size=kernel_size, 
                                bias=False, 
                                padding='same')
        
        self.norm = nn.LayerNorm(out_channels)
        self.activation = nn.GELU()

        
        
    def forward(self, x):
        """
        - downsample 
        - conv 
        - norm 
        - activation
        """
        
        x = self.downsample(x)
        
        x = self.conv1d(x)
        
        # norm by last axis.
        x = torch.transpose(x, -2, -1) 
        x = self.norm(x)
        x = torch.transpose(x, -2, -1) 
        
        x = self.activation(x)
        
        return x
    
class wav2vec_conv(nn.Module):
    """
    Extract some features from one time serias as raw speech.
    To do make it possibl;e to work with raw EEG electrode recording. 

    """
    def __init__(self,
                 n_inp_features=30,
                 channels = [32, 32, 32], 
                 kernel_sizes=[3, 3, 3],
                 strides=[2, 2, 2], 
                 dilations = [1, 1, 1]):
        
        super(wav2vec_conv, self).__init__()
        
        # freqs-electrodes  reducing to channels[0].
        # add additional layer
        channels = [n_inp_features] + channels
        
        self.model_depth = len(channels)-1
        # create downsample blcoks in Sequentional manner.
        self.downsample_blocks = nn.ModuleList([Conv_block(channels[i], 
                                                        channels[i+1], 
                                                        kernel_sizes[i],
                                                        stride=strides[i], 
                                                        dilation=dilations[i]) for i in range(self.model_depth)])
        
    def forward(self, x):
        """
        1. Encode  
        """
        batch, n_freq, time = x.shape
    
        # encode information
        for block  in self.downsample_blocks:
            x = block(x)
        return x

    
class Wav2Vec_aggregate(nn.Module):
    """
    Inpyt should be. 
    batch, n_ch, n_freq, time = x.shape
    
    model_embedding - should be work with [batch, n_freqs, time]
    
    Return [batch, n_ch, emb, time//stride] 
    """
    def __init__(self, model_embedding):
        
        super(Wav2Vec_aggregate, self).__init__()
        
        # create downsample blcoks in Sequentional manner.
        self.embedding = model_embedding
        
    def forward(self, x):
        """
        1. Apply for each channnels.  
        """
        batch, n_ch, n_freq, time = x.shape
        emb_list = [self.embedding(x[:, ch]) for ch in range(n_ch)]
        emb_list = torch.stack(emb_list, dim = 1)
        return emb_list
    
    
    
    
    
#### Transformer 


class Vanilla_transformer(nn.Module):
    """
    Vanilla transformer aggregate all 
    [batch, n_electrodes, embed_dim, time]
    
    Input of transformer is -> [batch, sequence_length, embed_dim]
    
    Return 
        [batch, 21]
    """
    def __init__(self,
                 n_electrodes = 64,
                 sequence_length = 128,
                 embed_dim=256,
                 n_roi=21,
                 num_heads=4,
                 mlp_ratio=2,
                 attn_dropout=0.1,
                 num_layers=1,                
                 mlp_dropout=0.1,
                ):
        super(Vanilla_transformer, self).__init__()
        self.n_electrodes = n_electrodes
        self.embed_dim = embed_dim
        self.sequence_length = sequence_length

        self.class_embed = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        
        # just add vector 
        self.pe = nn.Parameter(torch.zeros(1, n_electrodes*sequence_length + 1, embed_dim), requires_grad=True)
        
        # less parameters. factorises pe.

        self.pe_spatial = nn.Parameter(torch.zeros(1, n_electrodes, 1, embed_dim), requires_grad=True)
        self.pe_temporal = nn.Parameter(torch.zeros(1, 1, sequence_length , embed_dim), requires_grad=True)
        self.pe_cls =  nn.Parameter(torch.zeros(1, 1, self.embed_dim),  requires_grad=True)


        
        transformer_layer = nn.TransformerEncoderLayer(d_model=embed_dim, 
                                                            nhead=num_heads, 
                                                            dim_feedforward=embed_dim*mlp_ratio, 
                                                            dropout=attn_dropout, 
                                                            activation='relu', 
                                                            batch_first=True)
        
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)


        self.norm = nn.LayerNorm(embed_dim)
        # take our vector and make 21 prediction. 
        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*mlp_ratio),
            nn.Dropout(p=mlp_dropout), 
            nn.GELU(),
            nn.Linear(embed_dim*mlp_ratio, n_roi))
            
        
    def forward(self, x):
        """
        x.shape = > [batch, n_electrodes, embed_dim, time]
        
        """
        batch = x.shape[0]
        x = x.transpose(2, 3)
        x = x.reshape(batch, self.sequence_length * self.n_electrodes, self.embed_dim)
        
        # repeating like batch size and add seg lenght. 
        class_token = self.class_embed.expand(batch, -1, -1)
        x = torch.cat((class_token, x), dim=1)
        
        # add positional embeddings [batch, sequence_length, embed_dim]
        self.merge_pos_encoding()
        
        x += self.pe_merged
        x = self.transformer(x)
        
        # take first vector -> normalize -> mlp into 
        x_cls = x[:, 0]
        x_cls = self.mlp_head(self.norm(x_cls))
        return x_cls
    
    def merge_pos_encoding(self):
        """
        for calcualation.
        """
        self.pe_merged = self.pe_spatial + self.pe_temporal
        self.pe_merged = self.pe_merged.reshape(1, self.n_electrodes*self.sequence_length, self.embed_dim)
        self.pe_merged = torch.cat([self.pe_cls, self.pe_merged], dim= 1)
        
        
    
    

    
    

    
class Factorized_transformer(nn.Module):
    """
    Vanilla transformer aggregate all 
    [batch, n_electrodes, embed_dim, time]
    
    Input of transformer is -> [batch, sequence_length, embed_dim]
    
    
    
    Spatial transformer ( electrode aggregation ):
        [batch, n_electrodes, embed_dim, time] -> [batch, n_roi, embed_dim, time]
    Temporal transformer: (time aggregation) 
        [batch, n_roi, embed_dim, time] -> [batch, n_roi, embed_dim]
    Prediction
         [batch, n_roi, embed_dim] -> [batch, n_roi]
        
    
    
    
    Return 
        [batch, 21]
    """
    def __init__(self,
                 n_electrodes = 64,
                 sequence_length = 128,
                 embed_dim=256,
                 n_roi=21,
                 num_heads=4,
                 mlp_ratio=2,
                 attn_dropout=0.1,
                 num_layers_spatial=1,
                 num_layers_temporal=1,
                 mlp_dropout=0.1,
                ):
        super(Factorized_transformer, self).__init__()
        self.n_electrodes = n_electrodes
        self.embed_dim = embed_dim
        self.sequence_length = sequence_length
        self.n_roi = n_roi
        
                
        self.roi_tokens = nn.Parameter(torch.zeros(1, n_roi, embed_dim), requires_grad=True)
        self.time_tokens = nn.Parameter(torch.zeros(1, n_roi, 1, embed_dim), requires_grad=True) # use different reg tokent for each roi.
   

        self.pe_spatial = nn.Parameter(torch.zeros(1, n_roi + n_electrodes, embed_dim), requires_grad=True)
        self.pe_temporal = nn.Parameter(torch.zeros(1, 1 + sequence_length, embed_dim), requires_grad=True)
        
        
        spatial_layer = nn.TransformerEncoderLayer(d_model=embed_dim, 
                                                                nhead=num_heads, 
                                                                dim_feedforward=embed_dim*mlp_ratio, 
                                                                dropout=attn_dropout, 
                                                                activation='relu', 
                                                                batch_first=True)
        
        temporal_layer = nn.TransformerEncoderLayer(d_model=embed_dim, 
                                                            nhead=num_heads, 
                                                            dim_feedforward=embed_dim*mlp_ratio, 
                                                            dropout=attn_dropout, 
                                                            activation='relu', 
                                                            batch_first=True)
        
        self.spatial_transformer = nn.TransformerEncoder(spatial_layer, num_layers=num_layers_spatial)
        self.temporal_transformer = nn.TransformerEncoder(temporal_layer, num_layers=num_layers_temporal)



#         self.mlp_heads = nn.ModuleList([nn.Sequential(
#                                         nn.LayerNorm(embed_dim),
#                                         nn.Linear(embed_dim, embed_dim*mlp_ratio),
#                                         nn.Dropout(p=mlp_dropout), 
#                                         nn.GELU(),
#                                         nn.Linear(embed_dim*mlp_ratio, 1)
#                                         ) for i in range(self.n_roi)])
        
        self.mlp_heads = nn.ModuleList([nn.Sequential(
                                        nn.LayerNorm(embed_dim),
                                        nn.Linear(embed_dim, embed_dim//2),
                                        nn.Dropout(p=mlp_dropout), 
                                        nn.GELU(),
                                        nn.Linear(embed_dim//2, 1)
                                        ) for i in range(self.n_roi)])

        
    def forward(self, x):
        """
        x.shape = > [batch, n_electrodes, embed_dim, sequence_length]
        
        """
        
        batch = x.shape[0]
        x = x.permute(0, 3, 1, 2)
        
        # [batch, sequence_length, n_electrodes, embed_dim]
        # spatial transformer.
        
        x_spatial = x.reshape(batch*self.sequence_length, self.n_electrodes, self.embed_dim)
        
        roi_tokens = self.roi_tokens.expand(batch*self.sequence_length, -1, -1)
        x_spatial = torch.cat((roi_tokens, x_spatial), dim=1)
        x_spatial += self.pe_spatial
        x_spatial_transformed = self.spatial_transformer(x_spatial)
        
        roi_tokens = x_spatial_transformed[:, :self.n_roi]
        
        # OUTPUT: [batch*sequence_length, n_roi, embed_dim]
        ## temporal transfromer.
        # take only roi tokens 
        
        roi_tokens = roi_tokens.reshape(batch, self.sequence_length, self.n_roi, self.embed_dim)
        roi_tokens = roi_tokens.permute(0, 2, 1, 3)
        # [batch, n_roi, sequence_length, embed_dim]
        
        time_tokens = self.time_tokens.expand(batch, -1, -1, -1)
        roi_tokens = torch.cat((time_tokens, roi_tokens), dim=2)  # [batch, n_roi, 1+sequence_length, embed_dim]
        roi_tokens = roi_tokens.reshape(batch*self.n_roi, 1 + self.sequence_length, self.embed_dim)
        # [batch*n_roi, 1+sequence_length, embed_dim]
        
        
        roi_tokens += self.pe_temporal
        roi_tokens = self.temporal_transformer(roi_tokens)
        roi_tokens = roi_tokens[:, 0]
        roi_tokens = roi_tokens.reshape(batch, self.n_roi, self.embed_dim)
        
        # prediction 
        preds = []
        for roi in range(self.n_roi):
            res = self.mlp_heads[roi](roi_tokens[:, roi])
            preds.append(res)
        preds= torch.cat(preds, dim=1)
        
        return preds
    
    

    
    
def get_strided_func(stride):
    """
    stride parameters pooling 
    pool by last axis .
    """
    def get_strided_input(x):
        return x[..., ::stride]
    return get_strided_input


class Super_model(nn.Module):
    """
    Inpyt should be. 
    batch, n_ch, n_freq, time = x.shape
    
    window size - for inference pipeline. 
    How much EEG for prediction. 4096 for example.
    
    Return [batch, n_ch_out, emb, time//stride] 
    """
    def __init__(self, feature_extractor, transformer, window_size):
        
        super(Super_model, self).__init__()
        
        # create downsample blcoks in Sequentional manner.
        
        self.feature_extractor = feature_extractor
        self.transformer = transformer
        self.window_size = window_size
        
    def forward(self, x):
        """
        1. Apply for each channnels.  
        """
        batch, n_ch, n_freq, time = x.shape
        
        x_features = self.feature_extractor(x)
        x_out = self.transformer(x_features)
        
        return x_out
    
    
    
    
