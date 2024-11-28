import torch.nn as nn

class PositionWiseFNN(nn.Module):
    def __init__(self,d_model:int,d_ff:int=2048,dropout:float = 0.1):
        super().__init__()

        self.FNN = nn.Sequential(
            nn.Linear(d_model,d_ff),
            nn.ReLU(),
            nn.Linear(d_ff,d_model)
        )
        self.layer_nrom = nn.LayerNorm(d_model,eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        """
        x of shape : (batch_size,lengh_q,d_model)
        """
        #Pass through FFN layer
        ffn_x = self.FNN(x) #of shape : (batch_size,lengh_q,d_model)
        ffn_x = self.dropout(ffn_x)

        x = self.layer_nrom(ffn_x+x)

        return x