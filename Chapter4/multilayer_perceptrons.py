import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt

x = torch.arange(-8.0,8.0,0.1,requires_grad=True)
y1 = torch.relu(x)
y2 = torch.sigmoid(x)
y2.sum().backward()

d2l.plot(x.detach(),x.grad,'x','sigmoid(x)',figsize=(5,2.5))
plt.show()