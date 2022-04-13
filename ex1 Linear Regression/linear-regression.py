# coding=utf-8
"""
func: 一元线性回归
note: 使用梯度下降来实现线性回归; 归回方程(theta和x都是列向量): y' = theta_T * x; 损失函数采用均方误差：之差平方求和再除以2m
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 查看数据
def show_data(path, is_draw=False):
    data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
    print(f'data 数据样式:\n{data.shape}')
    print(f'数据样式:\n{data.head()}')
    print(f'基本数值:\n{data.describe()}')

    if is_draw:
        data.plot(kind='scatter', x='Population', y='Profit', figsize=(12, 8))
        plt.show()

    x, y = dataset(data)
    print(f'x 数据样式:\n{x.head()}')
    print(f'y 数据样式:\n{y.head()}')

    x = np.matrix(x.values)
    y = np.matrix(y.values)
    theta = np.matrix(np.array([0, 0]))
    print(f'{x.shape}, {y.shape}, {theta.shape}')


# 划分数据集
def dataset(data):
    cols = data.shape[1]
    x = data.iloc[:, 0: cols-1]
    # 最后一列
    y = data.iloc[:, cols-1: cols]

    # 插入一列1
    x.insert(0, 'Ones', 1)

    return x, y


# 计算损失函数; 损失函数采用均方误差：之差平方求和再除以2m
def compute_loss_func(x, y, theta):
    inner = np.power(((x * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(x))


# 梯度下降, alpha是学习率, epoch为迭代次数
def batch_gradient_decent(x, y, theta, alpha, epoch):
    # 参数副本
    temp = np.matrix(np.zeros(theta.shape))
    # 参数数目，ravel扁平化处理，返回一个连续的扁平数组，shape是1xn
    parameters = int(theta.ravel().shape[1])
    # 记录每次迭代的损失函数值
    cost = np.zeros(epoch)

    for i in range(epoch):
        # 差值，对于均方差来说，对每个权重参数求偏导的结果就是: (y' - y) * x
        error = (x * theta.T) - y

        for j in range(parameters):
            # 对应元素位置相乘, 这里计算了每个样本的偏导
            term = np.multiply(error, x[:, j])
            # 更新参数
            temp[0, j] = theta[0, j] - ((alpha / len(x)) * np.sum(term))

        theta = temp
        # 记录每次迭代之后的损失值
        cost[i] = compute_loss_func(x, y, theta)

    return theta, cost


# 训练
def train(path, is_draw=True):
    data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
    x, y = dataset(data)

    x = np.matrix(x.values)
    y = np.matrix(y.values)
    theta = np.matrix(np.array([0, 0]))

    alpha = 0.01
    epoch = 300

    print(f'cost: {compute_loss_func(x, y, theta)}')

    g, cost = batch_gradient_decent(x, y, theta, alpha, epoch)

    print(f'cost: {compute_loss_func(x, y, g)}')

    # 绘制拟合直线和学习曲线
    if is_draw:
        # 拟合直线
        x = np.linspace(data.Population.min(), data.Population.max(), 100)
        f = g[0, 0] + (g[0, 1] * x)

        plt.figure(figsize=(12, 4))
        ax = plt.subplot(1, 2, 1)
        ax.plot(x, f, 'r', label='Prediction')
        ax.scatter(data.Population, data.Profit, label='Training Data')
        # 0是默认，1234分别是从右上角逆时针位置
        ax.legend(loc=4)
        ax.set_xlabel('Population')
        ax.set_ylabel('Profit')
        ax.set_title('Predicted Profit vs. Population Size')
        # plt.show()

        # 学习曲线
        ax = plt.subplot(1, 2, 2)
        ax.plot(np.arange(epoch), cost, 'r')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Cost')
        ax.set_title('Error vs. Training Epoch')
        plt.show()


if __name__ == '__main__':
    file = 'ex1data1.txt'
    # 查看数据
    show_data(file)

    # 训练
    train(file)
