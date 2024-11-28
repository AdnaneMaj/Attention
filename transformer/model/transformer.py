import torch
import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder

def get_pad_mask(seq, pad_idx = 0):
    return (seq != pad_idx).unsqueeze(-2)

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    _, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask

class Transformer(nn.Module):
    def __init__(self,
                 d_model,
                 d_k,
                 d_v,
                 d_ff,
                 n_layers,
                 n_head,
                 max_len,
                 vocab_size,
                 pad_idx,
                 dropout,
                 tokenizer
                ):
        super().__init__()
        pad_idx = tokenizer.pad_token_id
        self.eos_idx = tokenizer.eos_token_id
        self.bos_idx = self.eos_idx #The MarianTokenizer don't have a special token for the begining of the sentence

        self.encoder = Encoder(d_model,d_k,d_v,d_ff,n_layers,n_head,max_len,vocab_size,pad_idx,dropout)
        self.decoder = Decoder(d_model,d_k,d_v,d_ff,n_layers,n_head,max_len,vocab_size,pad_idx,dropout)

        self.src_pad_idx, self.trg_pad_idx, = pad_idx, pad_idx
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.fn = nn.Linear(d_model,vocab_size,bias=False)

        #paramters initialisation
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

    def get_pad_mask(cls,seq, pad_idx = 0):
        return (seq != pad_idx).unsqueeze(-2)

    def get_subsequent_mask(lcs,seq):
        ''' For masking out the subsequent info. '''
        _, len_s = seq.size()
        subsequent_mask = (1 - torch.triu(
            torch.ones((len_s, len_s), device=seq.device), diagonal=1)).bool()
        return subsequent_mask

    def forward(self,src_seq,tgt_seq,src_mask=None,tgt_mask=None):
        """
        src_seq : (batch_size,src_lengh) of type int
        tgt_seq : (batch_size,trg_lengh) of type int
        """

        #get the masks if not provided
        src_mask = Transformer.get_pad_mask(src_seq, self.src_pad_idx) if src_mask is None else src_mask
        trg_mask = Transformer.get_pad_mask(tgt_seq, self.trg_pad_idx) & Transformer.get_subsequent_mask(tgt_seq) if tgt_mask is None else tgt_mask

        #Pass through the encoder and the decoder
        out_enc,_ = self.encoder(src_seq.clone(),src_mask) #(batch_size,src_lengh,d_model)
        out_dec,_,_ = self.decoder(tgt_seq.clone(),trg_mask,out_enc,src_mask) #(batch_size,src_lengh,d_model)

        #Linear to get the logits
        seq_logits = self.fn(out_dec) #(batch_size,src_lengh,vocab_size)

        return seq_logits
    

    def generate(self):
        pass

    