import torch

x=torch.randn(4, 2)
y = torch.view_as_complex(x)

print(y)
print(y.shape)