import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.RevIN import RevIN

class Model(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.individual = configs.individual
        # if self.individual:
        #     self.Linear = nn.ModuleList()
        #     for i in range(self.channels):
        #         self.Linear.append(nn.Linear(self.seq_len,self.pred_len))
        # else:
        #     self.Linear = nn.Linear(self.seq_len, self.pred_len)

        c_in = configs.enc_in

        if configs.data=="PEMS":
            self.revin=RevIN(c_in, affine=1)
        else:
            self.revin=RevIN(c_in, affine=1)

        self.Linear = nn.Linear(self.seq_len*11, self.pred_len)
        
        self.Flatten=nn.Flatten(start_dim=-2)
        self.std=configs.std

    def forward(self, x,y=None,z=None,w=None):
        # x: [Batch, Input length, Channel]
        # seq_last = x[:,-1:,:].detach()
        # x = x - seq_last

        means= x[:,0,0,:].unsqueeze(1).detach()
        x=x[:,1:,:,:]

        if self.std:
            stdev = torch.sqrt(torch.var(torch.sum(x,dim=1), dim=1, keepdim=True, unbiased=False) + 1e-5)
            # stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev.unsqueeze(1)


        x=x.permute(0,3,1,2)

        

        x=self.Flatten(x)

        # x=x.permute(0,2,1)
        # x = self.revin(x, 'norm')
        # x=x.permute(0,2,1)

        x = self.Linear(x)

        # x=x.permute(0,2,1)
        # x = self.revin(x, 'denorm')
        # x=x.permute(0,2,1)  

        x=x.permute(0,2,1)
        # x = x + seq_last


        L=self.pred_len
        if self.std:
            x = x * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))

        x = x + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))

        return x # [Batch, Output length, Channel]



