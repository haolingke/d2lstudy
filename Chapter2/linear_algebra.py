import torch

A = torch.tensor([[1,2,3],[4,5,6]])
x = torch.tensor([1,1,1])
ans = torch.mv(A,x)
print(ans)