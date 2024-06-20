import torch 
from torch import nn

X = torch.rand(20,10)
print(X)

net = nn.Sequential(nn.LazyLinear(256),nn.ReLU(),nn.Linear(256,10))

net(X)
print(net)
print(net[0].weight[1])