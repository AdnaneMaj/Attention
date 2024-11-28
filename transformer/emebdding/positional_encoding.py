import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self,max_len: int,d_model: int=512):
        """
        d_model : dimension of the model
        max_len : maximum sequence length
        """
        super(PositionalEncoding,self).__init__()

        self.encoding = torch.zeros(max_len,d_model)
        self.encoding.requires_grad = False #We don't need gradient

        #Position of the token
        pos = torch.arange(0,max_len)
        pos = pos.float().unsqueeze(dim=1) # (30,)=>(30,1)

        _2i = torch.arange(0,d_model,step=2)

        #compute positional encoding
        self.encoding[:,0::2] = torch.sin(pos/(10000**(_2i/d_model)))
        self.encoding[:,1::2] = torch.cos(pos/(10000**(_2i/d_model)))


    def forward(self,x):
        """
        Get positional encoding
        """
        batch_size,seq_len = x.size()

        return self.encoding[:seq_len,:]