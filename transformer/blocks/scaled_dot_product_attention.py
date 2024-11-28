import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    """
    Compute the scalled dot product attention for Q,K,V
    Note :
        Q,K,V are of dimension [batch_size,n_heads,lengh, dimension]
    """
    def __init__(self):
        super(ScaledDotProductAttention,self).__init__()

    def forward(self,q,k,v,mask=None):
        """
        k shape : batch_size,head,lengh_k,d_k
        q shape : batch_size,head,lengh_q,d_k
        v shape : batch_size,head,lengh_k,d_v
        """
        d_k = k.size(-1)

        #1. Dot product : Q.K^T
        k_t = k.transpose(2,3) # Of shape : (batch_size,head,d_k, lengh_k)
        att_score = torch.matmul(q,k_t)/torch.sqrt(torch.tensor(d_k)) #of shape : (batch_size,head,lengh_q, lengh_k)

        #2. Masking if wanted
        if mask is not None:
            #print("Att_shape",att_score.shape)
            #print("Att before",att_score)
            att_score = att_score.masked_fill(mask == 0,-1e9)
            #print("Att after",att_score)

        #3. Softmax
        att_score = F.softmax(att_score,dim=-1) 

        #4. Multiplay with value
        v = torch.matmul(att_score,v) #of shape : (batch_size,head,lengh_q, d_v)

        return v,att_score


