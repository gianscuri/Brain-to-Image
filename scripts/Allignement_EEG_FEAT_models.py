import torch
import torch.nn as nn
import torch.nn.functional as F

class FC(torch.nn.Module):

    def __init__(self,
                 input_dim: int = 100,
                 output_dim: int = 100):
        super(FC, self).__init__()

        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.linear(x)
        x = F.elu(x)
        x = F.dropout(x, p=0.5)
        x = self.linear(x)
        return x
    
class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim = 100,
        projection_dim=100,
        dropout=0.1
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x