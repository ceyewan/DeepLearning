import numpy as np
import matplotlib.pyplot as plt

# 准备数据
x_data = list()
y_data = list()


def prepare_data():
    x_data.extend([1.0, 2.0, 3.0])
    y_data.extend([2.0, 4.0, 6.0])


# 计算预测值
def forword(x, w):
    return x * w


# 计算损失值
def loss(y_pred, y):
    return (y_pred - y) ** 2


# 用于绘图
w_list = list()
mse_list = list()


# 穷举遍历每个可能的权重
def exhaustive():
    for w in np.arange(0.0, 4.1, 0.1):
        print("w:", w)
        # 损失总和
        l_sum = 0
        # 训练集每个 x 和 y
        for x_val, y_val in zip(x_data, y_data):
            # 预测值
            y_pred_val = forword(x_val, w)
            # 损失值
            loss_val = loss(y_pred_val, y_val)
            l_sum += loss_val
            print('\t', x_val, y_val, y_pred_val, loss_val)
        print('MSE=', l_sum / 3)
        w_list.append(w)
        mse_list.append(l_sum / 3)


# zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
# a = [1,2,3]
# b = [4,5,6]
# c = [4,5,6,7,8]
# zipped = zip(a,b)     # 打包为元组的列表
# [(1, 4), (2, 5), (3, 6)]
# zip(a,c)              # 元素个数与最短的列表一致
# [(1, 4), (2, 5), (3, 6)]
# zip(*zipped)          # 与 zip 相反，*zipped 可理解为解压，返回二维矩阵式
# [(1, 2, 3), (4, 5, 6)]

def show_image():
    plt.plot(w_list, mse_list)
    plt.ylabel('Loss')
    plt.xlabel('w')
    plt.show()


def main():
    prepare_data()
    exhaustive()
    show_image()


if __name__ == "__main__":
    main()
