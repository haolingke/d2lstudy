import torch
from torch import nn
from torch.nn import functional as F

x = torch.arange(4)
y = torch.zeros(4)

mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')



class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(10,256)
        self.out = nn.Linear(256,8)
    
    def forward(self,X):
        return self.out(F.relu(self.hidden(X)))
    
net = MLP()
X = torch.randn(size=(2, 10))
Y = net(X)

torch.save(net.state_dict(), 'mlp.params')
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
print(clone.eval())
Y_clone = clone(X)
print(Y_clone == Y)
