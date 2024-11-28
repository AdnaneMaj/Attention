import torch
import torch.nn as nn

from ..layers import DecoderLayer
from ..emebdding import TransformerEmbedding

class Decoder(nn.Module):
    """
    The encoder of the transofmer
    composed of N layers
    the ouput of each layer is the input of the second one
    """
    def __init__(self,d_model,d_k,d_v,d_ff,n_layers,n_head,max_len,vocab_size,pad_idx,dropout,TD:bool=False):
        super().__init__()
        self.TD = TD #True if it's a Transformer decoder only
        self.emb = TransformerEmbedding(d_model,max_len,vocab_size,dropout,pad_idx)
        self.dec_layers = nn.ModuleList([
            DecoderLayer(d_model,d_k,d_v,d_ff,n_head,dropout,TD)
            for _ in range(n_layers)
        ])

    def forward(self,trg_seq,trg_mask,out_enc=None,src_mask=None):
        """
        x : (batch_size,seq_lengh) of type int
        """
        #Embeding and positional encoding
        out_emb = self.emb(trg_seq) #(batch_size,seq_lengh,d_model)
        
        #Pass through the encoder layers
        att_list = []
        masked_att_list = []
        out_dec = out_emb
        for dec_layer in self.dec_layers:
            if not self.TD:
                out_dec,att_score,masked_att_score = dec_layer.forward(out_dec,trg_mask,out_enc,src_mask)
                att_list.append(att_score)
            else:
                out_dec,masked_att_score = dec_layer.forward(out_dec,trg_mask,out_enc,src_mask)
            masked_att_list.append(masked_att_score)
            
        return out_dec,att_list,masked_att_list #att_list will be simply empty in the case of a transformer only archittecture
    
    def generate(self,bos_token_id:int,tokenizer):
        context = torch.tensor([[bos_token_id]])
        for i in range(5):
            out = self.forward(context,None)
            next_token = out.max(dim=-1).indices
            context = torch.cat((context,next_token),dim=-1)
            
        return tokenizer.decode(context.flatten())