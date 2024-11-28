import torch.nn as nn

from ..layers import DecoderLayer
from ..emebdding import TransformerEmbedding
from .decoder import Decoder

class TransformerDecoder(nn.Module):
    """
    The encoder of the transofmer
    composed of N layers
    the ouput of each layer is the input of the second one
    """
    def __init__(self,d_model,d_k,d_v,d_ff,n_layers,n_head,max_len,vocab_size,pad_idx,dropout):
        super().__init__()
        self.pad_idx = pad_idx

        self.decoder = Decoder(d_model,d_k,d_v,d_ff,n_layers,n_head,max_len,vocab_size,pad_idx,dropout,TD=True)
        self.fn = nn.Linear(d_model,vocab_size,bias=False)

        #initialsie the parameters
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,trg_seq,trg_mask):
        """
        x : (batch_size,seq_lengh) of type int
        """
        #Pass through the decoder
        out_dec = self.decoder.forward(trg_seq,trg_mask)[0]
            
        #Linear to get the logits
        seq_logits = self.fn(out_dec) #(batch_size,src_lengh,vocab_size)

        return seq_logits