from torch import nn

class TokenEmbedding(nn.Embedding):
    """
    Token embedding
    """
    def __init__(self, vocab_size:int, d_model:int=512,pad_idx:int=0):
        super(TokenEmbedding,self).__init__(vocab_size, d_model,padding_idx=pad_idx)

