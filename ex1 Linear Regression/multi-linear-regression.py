# coding=utf-8
"""
func: 多元线性回归
note: 使用梯度下降来实现线性回归; 归回方程(theta和x都是列向量): y' = theta_T * x; 损失函数采用均方误差：之差平方求和再除以2m
      这里有两个变量，要多一步，归一化处理
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 查看数据
def show_data(path):
    data = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])
    print(f'data 数据样式:\n{data.shape}')
    print(f'数据样式:\n{data.head()}')
    print(f'基本数值:\n{data.describe()}')

    x, y = dataset(data)
    print(f'x 数据样式:\n{x.head()}')
    print(f'y 数据样式:\n{y.head()}')

    x = np.matrix(x.values)
    y = np.matrix(y.values)
    theta = np.matrix(np.array([0, 0, 0]))
    print(f'{x.shape}, {y.shape}, {theta.shape}')


# 划分数据集
def dataset(data):
    # 数据归一化
    data = (data - data.mean()) / data.std()

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
    data = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])
    x, y = dataset(data)

    x = np.matrix(x.values)
    y = np.matrix(y.values)
    theta = np.matrix(np.array([0, 0, 0]))

    alpha = 0.01
    epoch = 300

    print(f'cost: {compute_loss_func(x, y, theta)}')

    g, cost = batch_gradient_decent(x, y, theta, alpha, epoch)

    print(f'cost: {compute_loss_func(x, y, g)}')

    # 绘制拟合直线和学习曲线
    if is_draw:
        plt.figure(figsize=(12, 8))

        # 学习曲线
        ax = plt.subplot()
        ax.plot(np.arange(epoch), cost, 'r')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Cost')
        ax.set_title('Error vs. Training Epoch')
        plt.show()

    return g


# 正规方程
def normalEqn(path):
    data = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])
    x, y = dataset(data)

    x = np.matrix(x.values)
    y = np.matrix(y.values)

    theta = np.linalg.inv(x.T@x)@x.T@y  # X.T@X等价于X.T.dot(X)
    return theta


if __name__ == '__main__':
    file = 'ex1data2.txt'
    # 查看数据
    show_data(file)

    # 训练
    theta_train = train(file)
    print(theta_train)

    # 正规方程
    theta_normal = normalEqn(file)
    print(theta_normal)
