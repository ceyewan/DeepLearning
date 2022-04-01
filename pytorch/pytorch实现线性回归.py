import torch
# 1. prepare dataset

# 2. design model using Class

# 3. Construct loss and optimizer

# 4. Training cycle (forward,backward,update)

# prepare dataset
x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0]])


"""
our model class should be inherit from nn.Module, which is base class for all neural network modules.
member methods __init__() and forward() have to be implemented
class nn.linear contain two member Tensors: weight and bias
class nn.Linear has implemented the magic method __call__(),which enable the instance of the class can
be called just like a function.Normally the forward() will be called 
"""
# design model using class


class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        # (1,1)是指输入x和输出y的特征维度，这里数据集中的x和y的特征都是1维的
        # 该线性层需要学习的参数是w和b  获取w/b的方式分别是~linear.weight/linear.bias
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


model = LinearModel()


# construct loss and optimizer
# criterion = torch.nn.MSELoss(reduction='sum')
# 损失函数
criterion = torch.nn.MSELoss(size_average=False)
# 优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(10000):
    y_pred = model(x_data)  # forward: predict
    loss = criterion(y_pred, y_data)  # forward: loss
    # print(epoch, loss.item())

    # the grad computer by .backward() will be accumulated. so before backward, remember set the grad to zero
    loss.backward()  # backward
    optimizer.step()  # update w and b
    optimizer.zero_grad()

print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())

x_test = torch.tensor([[4.0]])
y_test = model.forward(x_test)
print('y_pred = ', y_test.item())
