# coding= utf-8
"""
func: 逻辑回归，实现二分类，添加正则化（即重新写损失函数计算和梯度求导）
note: 坐标点和类别，使用sigmoid函数对输出的结果做二分类
other: 如何画出边界线？
"""

import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt


# 查看数据
def show_data(data, theta, is_draw=False):
    print(f'data 数据样式:\n{data.shape}')
    print(f'数据样式:\n{data.head()}')
    print(f'基本数值:\n{data.describe()}')

    if is_draw:
        # 创建两个分数的散点图，并使用颜色编码来可视化，如果样本是正的（被接纳）或负的（未被接纳）
        positive = data[data['Admitted'].isin([1])]
        negative = data[data['Admitted'].isin([0])]

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
        ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
        ax.legend()
        ax.set_xlabel('Exam 1 Score')
        ax.set_ylabel('Exam 2 Score')
        plt.show()

    x, y = dataset(data)
    print(f'x 数据样式:\n{x.head()}')
    print(f'y 数据样式:\n{y.head()}')

    x = np.matrix(x.values)
    y = np.matrix(y.values)
    print(f'x: {x.shape}, y: {y.shape}, theta: {theta.shape}')


# 划分数据集
def dataset(data):
    cols = data.shape[1]
    x = data.iloc[:, 0: cols - 1]
    # 最后一列
    y = data.iloc[:, cols - 1: cols]

    # 插入一列1
    x.insert(0, 'Ones', 1)

    return x, y


# sigmoid 函数, S形函数, 数据规整到（0, 1）
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def test_sigmoid():
    nums = np.arange(-10, 10, step=1)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(nums, sigmoid(nums), 'r')
    plt.show()


# 梯度下降
def gradientReg(theta, x, y, lr):
    theta = np.matrix(theta)
    x = np.matrix(x)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(x * theta.T) - y

    for i in range(parameters):
        term = np.multiply(error, x[:, i])

        if i == 0:
            grad[i] = np.sum(term) / len(x)
        else:
            grad[i] = (np.sum(term) / len(x)) + ((lr / len(x)) * theta[:, i])

    return grad


# 正则化，重新写了损失函数
def costReg(theta, x, y, lr):
    theta = np.matrix(theta)
    x = np.matrix(x)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(x * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(x * theta.T)))
    reg = (lr / (2 * len(x))) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2))
    return np.sum(first - second) / len(x) + reg


# 训练
def train(data, theta):
    x, y = dataset(data)

    x = np.matrix(x.values)
    y = np.matrix(y.values)

    lr = 0.1

    print(f'cost1: {costReg(theta, x, y, lr)}')
    print(f'grad: {gradientReg(theta, x, y, lr)}')

    # 梯度优化，找到合适的theta，注意各个函数的参数顺序
    result = opt.fmin_tnc(func=costReg, x0=theta, fprime=gradientReg, args=(x, y, lr))
    print(f'result: {result}')

    print(f'cost2: {costReg(result[0], x, y, lr)}')

    return result[0]


# 预测函数
def predict(data, theta):
    x, y = dataset(data)

    # x是pd，values后是numpy，theta本身就是numpy
    x = np.matrix(x.values)
    y = np.matrix(y.values)
    theta = np.matrix(theta)

    probability = sigmoid(x * theta.T)
    output = [1 if p >= 0.5 else 0 for p in probability]

    correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(output, y)]
    accuracy = (sum(map(int, correct)) % len(correct))
    print('accuracy = {0}%'.format(accuracy))


if __name__ == '__main__':
    _file = 'ex2data2.txt'
    _data = pd.read_csv(_file, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
    _theta = np.zeros(3)

    # 查看数据
    show_data(_data, _theta, is_draw=False)

    # 测试sigmoid函数
    is_test = False
    if is_test:
        test_sigmoid()

    # 训练
    _theta = train(_data, _theta)

    # 预测
    predict(_data, _theta)
