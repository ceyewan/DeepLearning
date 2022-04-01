import numpy as np
import matplotlib.pyplot as plt
import torch

x_data = [1.0, 2.0, 3.0]
y_data = [1.0, 4.0, 9.0]

w1 = torch.tensor([1.0])  # w 的初始值为 1
w1.requires_grad = True  # 需要计算梯度
w2 = torch.tensor([1.0])
w2.requires_grad = True
b = torch.tensor([1.0])
b.requires_grad = True


def forward(x):
    return w1 * (x ** 2) + w2 * x + b


def loss(x, y):
    y_pred = forward(x)
    return (y - y_pred) ** 2


print('Predict (before training):', 4, forward(4).item())

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        # l是一个张量，tensor主要是在建立计算图 forward, compute the loss
        l = loss(x, y)
        # #  backward,compute grad for Tensor whose requires_grad set to True
        l.backward()
        print('\tgrad:', x, y, w1.grad.item(), w2.grad.item(), b.grad.item())
        w1.data = w1.data - 0.01 * w1.grad.data
        w2.data = w2.data - 0.01 * w2.grad.data
        b.data = b.data - 0.01 * b.grad.data
        # 清理，否则下次又会和这次的累加
        w1.grad.data.zero_()
        w2.grad.data.zero_()
        b.grad.data.zero_()
    print('Epoch:', epoch, l.item())

print('Predict (after training)', 4, forward(4).item())
