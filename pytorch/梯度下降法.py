import matplotlib.pyplot as plt

# 准备数据
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


# 随机猜一个权重
w = 1.0


# 求结果值
def forword(x):
    return x * w


# 所有元素计算好之后的损失值之和
def cost(xs, ys):
    cost = 0
    for x, y in zip(xs, ys):
        y_pred = forword(x)
        cost += (y_pred - y) ** 2
    return cost


# 求梯度，gd
def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (w * x - y)
    return grad


# 用于绘图
epoch_list = []
cost_list = []


print('predict (before training):', 4, forword(4))


# 训练 100 轮
for epoch in range(100):
    # 损失值之和
    cost_val = cost(x_data, y_data)
    # 梯度
    grad_val = gradient(x_data, y_data)
    # 梯度下降
    w -= 0.01 * grad_val  # 0.01 是学习率，可大可小
    print('epoch:', epoch, 'w=', w, 'loss=', cost_val)
    epoch_list.append(epoch)
    cost_list.append(cost_val)

print('predict (after training):', 4, forword(4))


# 绘制图像
plt.plot(epoch_list, cost_list)
plt.ylabel('cost')
plt.xlabel('epoch')
plt.show()
