import torch
import torch.nn as nn

from .scaled_dot_product_attention import ScaledDotProductAttention

class MultiHeadAttention(nn.Module):
    """
    Apply multi-head attention and Add&Norm
    """
    def __init__(self, d_model:int, d_k:int, d_v:int, n_head:int, dropout:float=0.1):
        super(MultiHeadAttention,self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.dot_prod_att = ScaledDotProductAttention()

        self.WV= nn.Linear(d_model,n_head*d_v,bias=False) #VWv
        self.WK = nn.Linear(d_model,n_head*d_k,bias=False) #KWk
        self.WQ = nn.Linear(d_model,n_head*d_k,bias=False) #QWk
        self.fc = nn.Linear(n_head*d_v,d_model,bias=False)

        self.dropout = nn.Dropout(dropout)
        self.layer_nrom = nn.LayerNorm(d_model,eps=1e-6)

    def forward(self,q,k,v,mask=None):
        
        residual = q #The query is considered to be the input x

        wq = self.WQ(q) #of shape (batch_size,lengh,n_head*d_k)
        wk = self.WK(k) #of shape (batch_size,lengh,n_head*d_k)
        wv = self.WV(v) #of shape (batch_size,lengh,n_head*d_v)

        batch_size,lengh_q,d_k,d_v,n_head = wq.size(0),wq.size(1),self.d_k,self.d_v,self.n_head

        # From (batch_size,lengh,n_head*d_k) to (batch_size,lengh,n_head,d_k) then (batch_size,n_head,lengh,d_k)
        wq = wq.view(batch_size,lengh_q,n_head,d_k).transpose(1,2)
        wk = wk.view(batch_size,lengh_q,n_head,d_k).transpose(1,2)
        wv = wv.view(batch_size,lengh_q,n_head,d_v).transpose(1,2) 

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q,att_score = self.dot_prod_att(wq,wk,wv,mask) #of shape (batch_size,n_head,lengh_q,d_v)

        # To (batch_size,lengh_q,n_head,d_v) then concat : (batch_size,lengh_q,n_head*d_v)
        q = q.transpose(1,2).contiguous().view(batch_size,lengh_q,-1) #of shape : (batch_size,lengh_q,n_head*d_v)
        q = self.dropout(self.fc(q))  #of shape : (batch_size,lengh_q,d_model)

        #Add & Norm
        q = self.layer_nrom(q+residual)

        return q,att_score