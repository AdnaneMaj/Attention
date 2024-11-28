import torch.nn as nn

from ..blocks import MultiHeadAttention,PositionWiseFNN

class DecoderLayer(nn.Module):
    """
    The decoder layer
            print("==>",result[0].shape,out_ffn.shape)
        print("==>",result[1].shape,att_score.shape)
        print("==>",result[2].shape,masked_att_score.shape)
        print("==>",result[3].shape)
    """
    def __init__(self, d_model:int, d_k:int, d_v:int, d_ff:int, n_head:int, dropout:float,TD:bool=False):
        super().__init__()
        self.TD = TD # If set to True, it mean this a decoder 
        self.masked_multi_head_att = MultiHeadAttention(d_model,d_k,d_v,n_head,dropout) #Self attention
        self.multi_head_att = MultiHeadAttention(d_model,d_k,d_v,n_head,dropout) if not TD else None #attention  (not needed in the case of Transformer decoder)
        self.fnn = PositionWiseFNN(d_model,d_ff,dropout)

    def forward(self,x,self_attn_mask,enc_out=None,dec_enc_attn_mask=None):
        """
        x : input of decoder (shifted right by one position)
            shape : (batch_size,seq_lenght,d_model)
        enc_out : output of the encoder
        """
        att_out,masked_att_score = self.masked_multi_head_att(x,x,x,self_attn_mask) #(batch_size,seq_lenght,d_model)
        if not self.TD:
            att_out,att_score = self.multi_head_att(enc_out,enc_out,att_out,dec_enc_attn_mask) #(batch_size,seq_lenght,d_model)
        out_ffn = self.fnn(att_out) #(batch_size,seq_lenght,d_model)

        return (out_ffn,att_score,masked_att_score) if not self.TD else (out_ffn,masked_att_score)