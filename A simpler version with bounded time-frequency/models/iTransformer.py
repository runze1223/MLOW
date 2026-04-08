__all__ = ['PatchTST']

# Cell
from typing import Callable, Optional
import torch

from layers.iTransformer_backbone import Interaction_backbone

import math

from layers.RevIN import RevIN
from layers.Embed import DataEmbedding_inverted
from torch import nn




class Model(nn.Module):
    def __init__(self, configs, **kwargs):
        super().__init__()   
        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len
        self.target_window=target_window
        
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model2 = configs.d_model2
        d_ff = configs.d_ff
        dropout2=configs.dropout2
        d_ff2=configs.d_ff2
        
        sr=context_window
        self.context_window=context_window


        if configs.data=="PEMS":
            self.revin=RevIN(c_in+1, affine=1)
        else:
            self.revin=RevIN(c_in+4, affine=1)

        self.std=configs.std

        self.rank=configs.rank
        
        self.model_interaction=Interaction_backbone(configs, context_window, target_window,d_model2,d_ff2, dropout2,n_heads,n_layers)

        self.Flatten=nn.Flatten(start_dim=-2)

        self.linear= nn.Linear(context_window*(self.rank+1),d_model2)


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):           # x: [Batch, Input length, Channel]

        _,_,_,b=x_enc.size()

        x=torch.cat([x_enc, x_mark_enc], -1)

        means= x[:,0,0,:].unsqueeze(1).detach()
        x=x[:,1:,:,:]

        if self.std:
            stdev = torch.sqrt(torch.var(torch.sum(x,dim=1), dim=1, keepdim=True, unbiased=False) + 1e-5)
            # stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev.unsqueeze(1)

        x = x.permute(0,3,1,2)

        x=self.Flatten(x)

        x=x.permute(0,2,1)
        x = self.revin(x, 'norm')
        x=x.permute(0,2,1)

        x=self.linear(x)

        x=x.permute(0,2,1)
        x = self.revin(x, 'denorm')
        x=x.permute(0,2,1)

        channel_mask=None
        residual=self.model_interaction(x,channel_mask)

        output=residual

        output=output.permute(0,2,1)

        L=self.target_window

        if self.std:
            output = output * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        
        output = output + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))

        output=output[:,:,:b]

        
        return output