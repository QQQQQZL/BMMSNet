import torch
from torch import nn
import torch.nn.functional as F



class Dilated_Conv_Model(nn.Module):        # CNN2加上了batch norm
    def __init__(self, feature_dim=1):
        super(Dilated_Conv_Model, self).__init__()
        self.name = 'Dilated_Conv_Model'

        
        
        self.conv_eeg = nn.Conv1d(64, 8, kernel_size=1, stride=1)

        self.dconv_eeg = nn.Sequential(nn.Conv1d(8, 16, kernel_size=3, dilation=3**0), nn.ReLU(),
                                       nn.Conv1d(16, 16, kernel_size=3, dilation=3**1), nn.ReLU(),
                                       nn.Conv1d(16, 16, kernel_size=3, dilation=3**2), nn.ReLU(),                                   
                                       )
        
        self.dconv_stimuli = nn.Sequential(nn.Conv1d(feature_dim, 16, kernel_size=3, dilation=3**0), nn.ReLU(),
                                       nn.Conv1d(16, 16, kernel_size=3, dilation=3**1), nn.ReLU(),
                                       nn.Conv1d(16, 16, kernel_size=3, dilation=3**2), nn.ReLU(),                                   
                                       )
        
        self.linear = nn.Linear(16*16, 1)

        self.softmax = nn.Softmax(-1)
       
    def forward(self, x):  	# x: [eeg(B, T, 64), stimuli0, stimuli1,...(B, T, 1)] 
        
        eeg = x[0]
        stimulus = x[1:]
        


        eeg = self.conv_eeg(eeg.permute((0, 2, 1)))
        eeg = self.dconv_eeg(eeg)                                       # (B, 16, T)
        stimulus_dconv = [self.dconv_stimuli(each.permute((0, 2, 1))) for each in stimulus]   # (B, 16, T)

        cos_score = [F.cosine_similarity(eeg.unsqueeze(1), each.unsqueeze(2), dim=-1) for each in stimulus_dconv]  # (B, 16, 16)
        
        cos_score_linear = [self.linear(each.flatten(1)) for each in cos_score]

        out = self.softmax(torch.cat(cos_score_linear, 1))
         

        return out
    

