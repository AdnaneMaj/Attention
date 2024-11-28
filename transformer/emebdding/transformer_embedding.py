import torch.nn as nn
from .positional_encoding import PositionalEncoding
from .token_embeddings import TokenEmbedding

class TransformerEmbedding(nn.Module):
    """
    The transformer embedding as a sum of token emebding and pos embedding with dropout
    """
    def __init__(self,d_model: int, max_len: int, vocab_size: int, drop_prob:float=0.1,pad_idx:int=0):
        """
        drop_p : dropout probability
        """
        super(TransformerEmbedding,self).__init__()
        self.pos_enc = PositionalEncoding(max_len,d_model)
        self.tok_emb = TokenEmbedding(vocab_size,d_model,pad_idx)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self,x):
        """
        Input of the encoder layers
        """
        return self.drop_out(self.tok_emb(x)+self.pos_enc(x))