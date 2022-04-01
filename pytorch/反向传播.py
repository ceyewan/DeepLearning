import numpy as np
import matplotlib.pyplot as plt
import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.tensor([1.0])  # w 的初始值为 1
w.requires_grad = True  # 需要计算梯度


# w 是一个张量，x也会转化为张量，最后返回的也是张量
def forward(x):
    return x * w


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
        print('\tgrad:', x, y, w.grad.item())
        w.data = w.data - 0.01 * w.grad.data
        # 清理，否则下次又会和这次的累加
        w.grad.data.zero_()
    print('Epoch:', epoch, l.item())

print('Predict (after training)', 4, forward(4).item())
