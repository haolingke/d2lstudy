# import torch
# x = torch.arange(4.0)
# x.requires_grad_(True)
# a = torch.add(x,1).detach()  #detach之后把a当常数用了
# b = torch.add(x,2)
# y = torch.mul(a,b)

# y.sum().backward()

# print("requires_grad: ", x.requires_grad, a.requires_grad, b.requires_grad, y.requires_grad)
# print("is_leaf: ", x.is_leaf, a.is_leaf, b.is_leaf, y.is_leaf)
# print("grad: ", x.grad, a.grad, b.grad, y.grad)

"""
  x (leaf, requires_grad=True)
  |
  |    +--- detach --- a (requires_grad=False)
  |   /
  +---> add(x, 1)
  |
  +---> add(x, 2) ------> b (requires_grad=True)
                          |
                          +------> mul(a, b) ------> y (requires_grad=True)

"""

#homework5
import torch
import matplotlib.pyplot as plt
x = torch.arange(0,7,0.1)
x1 = x.detach()   
"""
这一行很有必要，plt.plot(x,x.grad)会报错 
在PyTorch中，如果你尝试对一个需要梯度的张量调用 numpy() 方法，会抛出这个错误。为了避免这个问题，
你需要在将张量转换为NumPy数组之前调用 detach() 方法，从而使张量不再追踪梯度。
"""
x.requires_grad_(True)
fx = torch.sin(x)
fx.sum().backward()
plt.plot(x1,x.grad)
plt.show()
