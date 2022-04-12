# coding=utf-8
"""
func: 多分类，使用逻辑回归来识别手写数字（0到9），依旧是多元线性拟合
note: 数据是MATLAB的本机格式 ex3data1.mat
copyright: https://github.com/ydduong/Coursera-ML-AndrewNg-Notes/blob/master/code/ex3-neural%20network/ML-Exercise3.ipynb
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize


# 逻辑函数g(z) sigmoid, z是计算的结果，也就是theta和x的; 假设函数h(x)=g(z), z=theta*x
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 代价函数 J(theta)，二分类的逻辑回归，这个lr是lambda，控制正则大小的
def cost(theta, x, y, lr):
    theta = np.matrix(theta)
    x = np.matrix(x)
    y = np.matrix(y)
    # print(x.shape, theta.shape, y.shape)

    first = np.multiply(-y, np.log(sigmoid(x * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(x * theta.T)))
    # 正则项，除了第一个个权重，加上其他的平方，但是，为什么是lr来控制
    reg = (lr / (2 * len(x))) * np.sum(np.power(theta[:, 1: theta.shape[1]], 2))

    return np.sum(first - second) / len(x) + reg


# 使用循环的梯度下降，计算梯度
def gradient_loop(theta, x, y, lr):
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
            grad[i] = (np.sum(term) / len(x)) + (lr / len(x)) * theta[:, i]

    return grad


# 使用循环的梯度下降，向量化
def gradient(theta, x, y, lr):
    theta = np.matrix(theta)
    x = np.matrix(x)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    error = sigmoid(x * theta.T) - y

    grad = ((x.T * error) / len(x)).T + ((lr / len(x)) * theta)
    grad[0, 0] = np.sum(np.multiply(error, x[:, 0])) / len(x)

    return np.array(grad).ravel()


# 多分类器, 利用数据，训练出不同标签的预测参数
def one_vs_all(x, y, num_labels, lr):
    rows = x.shape[0]
    params = x.shape[1]

    all_theta = np.zeros((num_labels, params + 1))
    x = np.insert(x, 0, values=np.ones(rows), axis=1)

    for i in range(1, num_labels + 1):
        theta = np.zeros((params + 1))
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i, (rows, 1))

        find_mini = minimize(fun=cost, x0=theta, args=(x, y_i, lr), method='TNC', jac=gradient)
        all_theta[i - 1, :] = find_mini.x

    return all_theta


if __name__ == "__main__":
    _lr = 0.3

    # 加载数据，图像是20x20的灰度图像，也就是400
    _data = loadmat('ex3data1.mat')
    print(_data['X'].shape, _data['y'].shape)

    # 数据向量化
    _rows = _data['X'].shape[0]
    _params = _data['X'].shape[1]

    _all_theta = np.zeros((10, _params + 1))

    _x = np.insert(_data['X'], 0, values=np.ones(_rows), axis=1)
    _theta = np.zeros(_params + 1)

    _y_0 = np.array([1 if label == 0 else 0 for label in _data['y']])
    _y_0 = np.reshape(_y_0, (_rows, 1))

    print(_y_0.shape, _x.shape, _theta.shape, _all_theta.shape, np.unique(_data['y']))

    # 测试 sigmoid
    is_draw = False
    if is_draw:
        nums = np.arange(-10, 10, step=1)
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(nums, sigmoid(nums), 'r')
        plt.show()

    # 测试 cost
    print(f'cost: {cost(_theta, _x, _y_0, _lr)}')

    # 测试 gradient
    grad = gradient_loop(_theta, _x, _y_0, _lr)
    print(f'gradient: {len(grad)}, {grad[0]}')
    grad = gradient(_theta, _x, _y_0, _lr)
    print(f'gradient: {len(grad)}, {grad[0]}')

    # 测试 one_vs_all
    _all_theta = one_vs_all(_data['X'], _data['y'], 10, _lr)

    # 测试 标签0的训练结果
    _theta = _all_theta[0, :]
    print(f'cost: {cost(_theta, _x, _y_0, _lr)}')
    grad = gradient_loop(_theta, _x, _y_0, _lr)
    print(f'gradient: {len(grad)}, {grad[0]}')

    # 预测
    # 前向计算
    _all_theta = np.matrix(_all_theta)
    h = sigmoid(_x * _all_theta.T)

    # h is 5000x10, h_argmax is 5000x1

    # create array of the index with the maximum probability and plus 1
    h_argmax = np.argmax(h, axis=1) + 1

    correct = [1 if a == b else 0 for (a, b) in zip(h_argmax, _data['y'])]
    accuracy = (sum(map(int, correct)) / float(len(correct)))
    print('accuracy = {0}%'.format(accuracy * 100))
