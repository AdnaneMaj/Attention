import torch.nn as nn

from ..blocks import MultiHeadAttention,PositionWiseFNN

class EncoderLayer(nn.Module):
    """
    The encoder layer :
        - Multi-head attetnion + Add&Norm
        - FFN + Add&Norm
    """
    def __init__(self,d_model:int, d_k:int, d_v:int, d_ff:int, n_head:int, dropout:float):
        super().__init__()
        self.multi_head_att = MultiHeadAttention(d_model,d_k,d_v,n_head,dropout)
        self.ffn = PositionWiseFNN(d_model,d_ff,dropout)

    def forward(self,x,self_attn_mask=None):
        """
        input :
            x : (batch_size,seq_lengh, d_model)
        """
        #q,v,k will be learned with Wq,Wv,Wk
        out_att,att_score = self.multi_head_att(x,x,x,self_attn_mask) #(batch_size,seq_lengh,d_model)

        out_ffn = self.ffn(out_att) #(batch_size,seq_lengh,d_model)

        return out_ffn,att_score