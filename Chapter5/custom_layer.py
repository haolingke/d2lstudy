import torch
from torch import nn
import torch.nn.functional as F

class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,X):
        return X - X.mean()
    
class Mylayer(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        self.weight =  nn.Parameter(torch.randn(in_feature,out_feature))
        self.bias = nn.Parameter(torch.randn(out_feature,))
    
    def forward(self,X):
        linear = torch.matmul(X,self.weight.data) + self.bias.data
        return F.relu(linear)





layer = CenteredLayer()
layer(torch.FloatTensor([1,23,4]))

net = nn.Sequential(nn.Linear(8,8),CenteredLayer())
X = torch.rand(4,8)
Y = net(X)


linear = Mylayer(5,3)
print(linear(torch.rand(2,5)))