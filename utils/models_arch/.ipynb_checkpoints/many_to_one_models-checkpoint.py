import torch

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader, Subset
from pytorch_model_summary import summary
import numpy as np
import wandb

import numpy as np
class SimpleNet(nn.Module):
    """
    Usage example 
    model = SimpleNet(in_channels=30, out_channels=21, window_size=4096)
    print(summary(model, torch.zeros(4, 30, 4096)))
    """
    def __init__(self, in_channels, out_channels, 
                window_size, 
                DOWNSAMPLING =  32,
                HIDDEN_CHANNELS = 16,
                FILTERING_SIZE = 51,
                ENVELOPE_SIZE = 51, 
                BIDIRECTIONAL = True):
        super(self.__class__,self).__init__()

        self.DOWNSAMPLING =  DOWNSAMPLING
        self.HIDDEN_CHANNELS = HIDDEN_CHANNELS
        self.FILTERING_SIZE = FILTERING_SIZE
        self.ENVELOPE_SIZE = ENVELOPE_SIZE
    
        self.BIDIRECTIONAL = BIDIRECTIONAL

        assert FILTERING_SIZE % 2 == 1, "conv weights must be odd"
        assert ENVELOPE_SIZE % 2 == 1, "conv weights must be odd"
        


        self.window_size = window_size
        self.final_out_features = self.window_size // self.DOWNSAMPLING * self.HIDDEN_CHANNELS
        print(self.final_out_features)
        self.final_out_features *= 2 if self.BIDIRECTIONAL else 1
        print(self.final_out_features)
        
        
        assert window_size > self.FILTERING_SIZE
        assert window_size > self.ENVELOPE_SIZE

        self.unmixing_layer = nn.Conv1d(in_channels, self.HIDDEN_CHANNELS, 1)
        self.unmixed_channels_batchnorm = nn.BatchNorm1d(self.HIDDEN_CHANNELS,
                                                         affine=False)
        
        self.detector = self._create_envelope_detector(self.HIDDEN_CHANNELS)
        self.features_batchnorm = nn.BatchNorm1d(self.final_out_features, affine=False)
        
        self.avg_pool = nn.AvgPool1d(self.DOWNSAMPLING, stride=self.DOWNSAMPLING)
        self.max_pool = nn.MaxPool1d(self.DOWNSAMPLING, stride=self.DOWNSAMPLING)

        
        self.lstm = nn.LSTM(self.HIDDEN_CHANNELS, self.HIDDEN_CHANNELS, 
                            num_layers=2,
                            batch_first=True,
                            bidirectional=self.BIDIRECTIONAL)
        
        n_features = self.HIDDEN_CHANNELS
        n_features *= 2 if self.BIDIRECTIONAL else 1
        self.bn_last = nn.BatchNorm1d(n_features, affine=False)
        self.fc_last = nn.Linear(n_features, out_channels)
        
        self.fc_layer = nn.Linear(self.final_out_features, out_channels)

    def _create_envelope_detector(self, in_channels):  
        # 1. Learn band pass flter
        # 2. Centering signals
        # 3. Abs
        # 4. low pass to get envelope
        
        envelope_detect_model = nn.Sequential(
                                    nn.Conv1d(in_channels, in_channels,
                                              kernel_size=self.FILTERING_SIZE,
                                              bias=False,  groups=in_channels,
                                              padding='same'),
                                    nn.BatchNorm1d(in_channels, affine=False),
                                    nn.LeakyReLU(-1), 
                                    nn.Conv1d(in_channels, in_channels,
                                              kernel_size=self.ENVELOPE_SIZE,
                                              groups=in_channels, padding='same'))
        
        return envelope_detect_model

    def forward(self, x):
        """
        1. Spatial filter -> BN
        2. SepConv -> BN -> SepConv
        3. LSTM -> flat -> BN
        4. Linear 
        """
        
        unmixed_channels = self.unmixing_layer(x)
        
        unmixed_channels_scaled = self.unmixed_channels_batchnorm(unmixed_channels)
        
        detected_envelopes = self.detector(unmixed_channels_scaled)
        
        features = self.max_pool(detected_envelopes)
#         features = detected_envelopes[:, :, ::self.DOWNSAMPLING].contiguous() # batch, ch, time

        # batch, ch, time -> batch, time, ch 
        features =  features.transpose(1, 2)

        #  batch, time, ch -> (batch, time, D*hid) ,(D*n_layers, batch, hid, (D*n_layers, batch, hid))
        features, (h_n, c_n) = self.lstm(features)

        last_hidden_vector = features[:, -1]
        output = self.bn_last(last_hidden_vector)
        output = self.fc_last(output)
        
        
#         # batch, time, D*hid > batch, D*hid, time -> 
#         features = features.transpose(1, 2)
#         features = features.reshape((features.shape[0], -1)) # flatten
  
#         self.features_scaled = self.features_batchnorm(features)
#         output = self.fc_layer(self.features_scaled)
        
        return output