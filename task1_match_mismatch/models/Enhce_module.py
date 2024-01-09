import torch
from torch import nn
import torch.nn.functional as F
from task1_match_mismatch.models.loss import pearson_torch
from task1_match_mismatch.models.Transformer_torch import TransformerSentenceEncoderLayer
import math
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.transpose(0, 1).unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :, :x.size(2)]
        return self.dropout(x)


class EEG_enhce_module(nn.Module):   #可以将kernel_size作为参数了
    def __init__(self):
        super(EEG_enhce_module, self).__init__()
        self.name = 'EEG_enhce_module'
        

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        self.encoder.append(nn.Sequential(nn.Conv2d(1, 4, kernel_size = (3, 5), padding=(1,2), stride=(1,1)), nn.Dropout(0.2),nn.BatchNorm2d(4), nn.Sigmoid()))
        self.encoder.append(nn.Sequential(nn.Conv2d(4, 4, kernel_size = (3, 5), padding=(1,2), stride=(1,1)), nn.Dropout(0.2),nn.BatchNorm2d(4), nn.Sigmoid()))
        self.encoder.append(nn.Sequential(nn.Conv2d(4, 8, kernel_size = (3, 5), padding=(1,2), stride=(2,2)), nn.Dropout(0.2),nn.BatchNorm2d(8), nn.Sigmoid()))
        self.encoder.append(nn.Sequential(nn.Conv2d(8, 8, kernel_size = (3, 5), padding=(1,2), stride=(1,1)), nn.Dropout(0.2),nn.BatchNorm2d(8), nn.Sigmoid()))
        self.encoder.append(nn.Sequential(nn.Conv2d(8, 16, kernel_size = (3, 5), padding=(1,2), stride=(2,2)), nn.Dropout(0.2),nn.BatchNorm2d(16), nn.Sigmoid()))
        self.encoder.append(nn.Sequential(nn.Conv2d(16, 16, kernel_size = (3, 5), padding=(1,2), stride=(1,1)), nn.Dropout(0.2),nn.BatchNorm2d(16), nn.Sigmoid()))
        self.encoder.append(nn.Sequential(nn.Conv2d(16, 32, kernel_size = (3, 5), padding=(1,2), stride=(2,2)), nn.Dropout(0.2),nn.BatchNorm2d(32), nn.Sigmoid()))
        self.encoder.append(nn.Sequential(nn.Conv2d(32, 32, kernel_size = (3, 5), padding=(1,2), stride=(1,1)), nn.Dropout(0.2),nn.BatchNorm2d(32), nn.Sigmoid()))
        self.decoder.append(nn.Sequential(nn.ConvTranspose2d(32, 32, kernel_size = (3, 5), padding=(1,2), stride=(1,1)), nn.Dropout(0.2),nn.BatchNorm2d(32), nn.Sigmoid()))
        self.decoder.append(nn.Sequential(nn.ConvTranspose2d(64, 16, kernel_size = (3, 5), padding=(1,2), stride=(2,2), output_padding=(1,1)), nn.Dropout(0.2),nn.BatchNorm2d(16), nn.Sigmoid()))
        self.decoder.append(nn.Sequential(nn.ConvTranspose2d(32, 16, kernel_size = (3, 5), padding=(1,2), stride=(1,1)), nn.Dropout(0.2),nn.BatchNorm2d(16), nn.Sigmoid()))
        self.decoder.append(nn.Sequential(nn.ConvTranspose2d(32, 8, kernel_size = (3, 5), padding=(1,2), stride=(2,2), output_padding=(1,1)), nn.Dropout(0.2),nn.BatchNorm2d(8), nn.Sigmoid()))
        self.decoder.append(nn.Sequential(nn.ConvTranspose2d(16, 8, kernel_size = (3, 5), padding=(1,2), stride=(1,1)), nn.Dropout(0.2),nn.BatchNorm2d(8), nn.Sigmoid()))
        self.decoder.append(nn.Sequential(nn.ConvTranspose2d(16, 4, kernel_size = (3, 5), padding=(1,2), stride=(2,2), output_padding=(1,1)), nn.Dropout(0.2),nn.BatchNorm2d(4), nn.Sigmoid()))
        self.decoder.append(nn.Sequential(nn.ConvTranspose2d(8, 4, kernel_size = (3, 5), padding=(1,2), stride=(1,1)), nn.Dropout(0.2),nn.BatchNorm2d(4), nn.Sigmoid()))
        self.decoder.append(nn.Sequential(nn.ConvTranspose2d(8, 1, kernel_size = (3, 5), padding=(1,2), stride=(1,1)), nn.Dropout(0.2),nn.BatchNorm2d(1), nn.Sigmoid()))
        self.transformer = nn.Sequential(PositionalEncoding(32*8), TransformerSentenceEncoderLayer(embedding_dim= 32*8, ffn_embedding_dim=128, num_attention_heads=4),)


    def forward(self, x):
        
        encoder_out = []
        for i in range(8):
            x = self.encoder[i](x)
            encoder_out.append(x)

        B, C1, C2, T = x.shape
        # x = x.reshape((B, C1*C2, T)).permute((0, 2, 1))
        # x = self.gru(x)[0].permute((0, 2, 1)).reshape((B, C1, C2, T))

        x = x.reshape((B, C1*C2, T))
        x = self.transformer(x).reshape((B, C1, C2, T))
        x = self.decoder[0](x) 
        for i in range(7):
            x = self.decoder[i+1](torch.cat((x, encoder_out[6-i]), 1))

        return x
    
