import torch
import torch.nn as nn
import torch.nn.functional as F

class FC(torch.nn.Module):

    def __init__(self,
                 input_dim: int = 100,
                 output_dim: int = 1654):
        super(FC, self).__init__()

        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.linear(x)
        return x