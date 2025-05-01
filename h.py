import  torch

a = torch.zeros((1,1,1,1,3))
print(a.squeeze(dim = 0))