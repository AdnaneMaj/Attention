import torch.nn as nn

from ..layers import EncoderLayer
from ..emebdding import TransformerEmbedding

class Encoder(nn.Module):
    """
    The encoder of the transofmer
    composed of N layers
    the ouput of each layer is the input of the second one
    """
    def __init__(self,d_model,d_k,d_v,d_ff,n_layers,n_head,max_len,vocab_size,pad_idx,dropout):
        super().__init__()
        self.emb = TransformerEmbedding(d_model,max_len,vocab_size,dropout,pad_idx)
        self.enc_layers = nn.ModuleList([
            EncoderLayer(d_model,d_k,d_v,d_ff,n_head,dropout)
            for _ in range(n_layers)
        ])


    def forward(self,src_seq,src_mask=None):
        """
        src_seq : (batch_size,seq_lengh) of type int
        """
        #Embeding and positional encoding
        out_emb = self.emb(src_seq) #(batch_size,seq_lengh,d_model)
        
        #Pass through the encoder layers
        att_list = []
        out_enc = out_emb #initialisaation
        for enc_layer in self.enc_layers:
            out_enc,att_score = enc_layer(out_enc,src_mask)
            att_list.append(att_score)

        return out_enc,att_list