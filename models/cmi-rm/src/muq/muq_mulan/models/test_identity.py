from torch import nn
import torch

class Indentity(nn.Module):
    def forward(self, *args):
        return args
model = Indentity(
)

x = torch.randn(2, 3, 4)
y = torch.randn(2, 3)
x, y = model(x, y)
print(x.shape)
print(y.shape)