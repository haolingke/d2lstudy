import torch
from torch import nn
from d2l import torch as d2l

def corr2d(X,K): #@save
    """计算二维互相关运算"""
    h,w = K.shape
    Y = torch.zeros(X.shape[0]-h+1,X.shape[1]-w+1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j] = (X[i:i+h,j:j+w]*K).sum()
    return Y

# X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
# K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
# print(corr2d(X, K))

X = torch.ones((6, 8))
X[:, 2:6] = 0
K = torch.tensor([[1.0, -1.0]])
Y = corr2d(X, K)

conv2d = nn.Conv2d(1,1,kernel_size=(1,2),bias=False)
X = X.reshape(1,1,6,8)
Y = Y.reshape(1,1,6,7)

lr = 3e-2
for i in range(10):
    Y_hat = conv2d(X)
    loss = (Y_hat-Y)**2
    conv2d.zero_grad()
    loss.sum().backward()
    conv2d.weight.data[:] -= lr*conv2d.weight.grad
    if (i+1)%2 == 0:
        print(f'epoch{i+1},loss{loss.sum():.3f}')

K_hat = conv2d.weight.data.reshape((1,2))
print(K_hat)