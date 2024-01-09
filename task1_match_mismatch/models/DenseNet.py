import torch
from torch import nn
import torch.nn.functional as F
from task1_match_mismatch.models.loss import pearson_torch
from BMMSNet.task1_match_mismatch.models.Enhce_module import EEG_enhce_module

class DenseBlock5(nn.Module):   #可以将kernel_size作为参数了
    def __init__(self, in_channels, growth_rate, num_layers, kernel_size=5, padding='same', dropout=0.2, act='relu'):
        super(DenseBlock5, self).__init__()
        self.layers = nn.ModuleList()
        if padding=='same':
            self.padding = (kernel_size//2, kernel_size//2)
        else:
            self.padding=padding
        

        for i in range(num_layers):
            if act == 'relu':
                self.layers.append(nn.Sequential(

                    nn.Conv1d(in_channels + i * growth_rate, growth_rate, kernel_size=kernel_size, padding=0, dilation=1, bias=True),
                    nn.Dropout(dropout),
                    nn.BatchNorm1d(growth_rate),
                    nn.ReLU()
                ))
            elif act == 'prelu':
                self.layers.append(nn.Sequential(

                    nn.Conv1d(in_channels + i * growth_rate, growth_rate, kernel_size=kernel_size, padding=0, dilation=1, bias=True),
                    nn.Dropout(dropout),
                    nn.BatchNorm1d(growth_rate),
                    nn.PReLU()
                ))
            elif act == 'sigmoid':
                self.layers.append(nn.Sequential(

                    nn.Conv1d(in_channels + i * growth_rate, growth_rate, kernel_size=kernel_size, padding=0, dilation=1, bias=True),
                    nn.Dropout(dropout),
                    nn.BatchNorm1d(growth_rate),
                    nn.Sigmoid()
                ))
    def forward(self, x):
        
        features = [x]
        for layer in self.layers:
            x = layer(F.pad(torch.cat(features, dim=1), self.padding, value=0))
            features.append(x)
        return torch.cat(features, dim=1),  features[1:], x 
    
    

class BMMSNet(nn.Module):        # 每层的输出都拿出来
    def __init__(self, feature_dim=1):
        super(BMMSNet, self).__init__()
        self.name = 'BMMSNet'
        self.layers=3
        self.enhce_module = EEG_enhce_module()
        self.dense_eeg1 = nn.Sequential(nn.Conv1d(128, 32, kernel_size=5, dilation=3**0, padding='same'), nn.PReLU(),
                                        DenseBlock5(32, 16, 10, act='prelu'), )
       

        self.dense_stimulus1 = nn.Sequential(nn.Conv1d(feature_dim, 32, kernel_size=5, dilation=3**0, padding='same'), nn.PReLU(),
                                        DenseBlock5(32, 16, 10, act='prelu'), )
       
        self.linear = nn.Linear(160, 1)
        self.softmax = nn.Softmax(-1)       
              
    def forward(self, input):  	# x: [eeg(B, T, 64), stimuli, stimuli(B, T, feature_dim)] 
        eeg = input[0].permute((0, 2, 1))
        eeg_enhce = self.enhce_module(eeg.unsqueeze(1)).squeeze(1)
        stimulus_all = input[1:]
        eeg = self.dense_eeg1(torch.cat((eeg, eeg_enhce),1))[1]
        eeg.reverse()
        eeg = torch.cat(eeg, 1)
        

        stimulus_out = []
        for stimulus in stimulus_all:
            stimulus = self.dense_stimulus1(stimulus.permute((0, 2, 1)))[1]
            
            stimulus_out.append(torch.cat(stimulus,1))
        
        cos_score = [pearson_torch(eeg, each, axis=-1) for each in stimulus_out]

        
        cos_score_linear = [self.linear(each) for each in cos_score]

        out = self.softmax(torch.cat(cos_score_linear, 1))
        
        return out 
