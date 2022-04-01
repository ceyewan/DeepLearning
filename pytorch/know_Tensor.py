import torch
a = torch.tensor([1.0])
a.requires_grad = True
print(a)
print(a.data)
print(a.data.item())
print(a.type())
print(a.data.type())
print(a.grad)
print(type(a.grad))

# tensor([1.], requires_grad=True)
# tensor([1.])
# 1.0
# torch.FloatTensor
# torch.FloatTensor
# None
# <class 'NoneType'>
