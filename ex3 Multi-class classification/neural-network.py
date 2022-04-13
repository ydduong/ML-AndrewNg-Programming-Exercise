# coding=utf-8
"""
func: 使用反向传播的前馈神经网络，数字图片分类
node: 将通过反向传播算法实现神经网络成本函数和梯度计算的非正则化和正则化版本
      还将实现随机权重初始化和使用网络进行预测的方法
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn.preprocessing import OneHotEncoder


# sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 前向传播 (400 + 1) -> (25 + 1) -> (10)，两层，两个权重
def forward(x, theta1, theta2):
    m = x.shape[0]

    # a1 = x add x0, x is 5000x400，加一个bias
    a1 = np.insert(x, 0, values=np.ones(m), axis=1)

    # 第二层
    z2 = a1 * theta1.T
    a2 = sigmoid(z2)
    a2 = np.insert(a2, 0, values=np.ones(m), axis=1)

    # 第三层
    z3 = a2 * theta2.T
    h = sigmoid(z3)

    return a1, z2, a2, z3, h


# 代价函数 形似逻辑回归中的代价函数
def cost(params, input_size, hidden_size, num_labels, x, y, lr):
    m = x.shape[0]
    x = np.matrix(x)
    y = np.matrix(y)

    # get theta: (400 + 1) -> (25 + 1) -> (10). theta1: 25x401. theta2: 10x26
    theta1 = np.matrix(np.reshape(params[: hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

    # 前向计算
    a1, z2, a2, z3, h = forward(x, theta1, theta2)

    # 计算各个样本的误差
    j = 0
    for i in range(m):
        first_term = np.multiply(-y[i, :], np.log(h[i, :]))
        second_term = np.multiply((1 - y[i, :]), np.log(1 - h[i, :]))
        j += np.sum(first_term - second_term)

    return j / m


# 代价函数 形似逻辑回归中的代价函数， 添加正则化
def cost_reg(params, input_size, hidden_size, num_labels, x, y, lr):
    m = x.shape[0]
    x = np.matrix(x)
    y = np.matrix(y)

    # get theta: (400 + 1) -> (25 + 1) -> (10). theta1: 25x401. theta2: 10x26
    theta1 = np.matrix(np.reshape(params[: hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

    # 前向计算
    a1, z2, a2, z3, h = forward(x, theta1, theta2)

    # 计算各个样本的误差
    j = 0
    for i in range(m):
        first_term = np.multiply(-y[i, :], np.log(h[i, :]))
        second_term = np.multiply((1 - y[i, :]), np.log(1 - h[i, :]))
        j += np.sum(first_term - second_term)

    j = j / m

    # 正则化 不会对偏置，操作
    j += float(lr / (2 * m)) * (np.sum(np.power(theta1[:, 1:], 2)) + np.sum(np.power(theta2[:, 1:], 2)))

    return j


# sigmoid 求导
def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), 1 - sigmoid(z))


# 反向传播
def back_propagation(params, input_size, hidden_size, num_labels, x, y, lr):
    m = x.shape[0]
    x = np.matrix(x)
    y = np.matrix(y)

    # get theta: (400 + 1) -> (25 + 1) -> (10). theta1: 25x401. theta2: 10x26
    theta1 = np.matrix(np.reshape(params[: hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

    # 前向计算
    a1, z2, a2, z3, h = forward(x, theta1, theta2)

    # 计算各个样本的误差
    j = 0
    delta1 = np.zeros(theta1.shape)  # (25, 401)
    delta2 = np.zeros(theta2.shape)  # (10, 26)

    for i in range(m):
        first_term = np.multiply(-y[i, :], np.log(h[i, :]))
        second_term = np.multiply((1 - y[i, :]), np.log(1 - h[i, :]))
        j += np.sum(first_term - second_term)

    j = j / m

    # 正则化 不会对偏置，操作
    j += float(lr / (2 * m)) * (np.sum(np.power(theta1[:, 1:], 2)) + np.sum(np.power(theta2[:, 1:], 2)))

    # 计算一步梯度
    for t in range(m):
        a1t = a1[t, :]  # (1, 401)
        z2t = z2[t, :]  # (1, 25)
        a2t = a2[t, :]  # (1, 26)
        ht = h[t, :]    # (1， 10)
        yt = y[t, :]    # (1， 10)

        d3t = ht - yt   # (1， 10)

        z2t = np.insert(z2t, 0, values=np.ones(1))  # (1, 26)
        d2t = np.multiply(d3t * theta2, sigmoid_gradient(z2t))  # (1, 26)

        delta1 += (d2t[:, 1:]).T * a1t
        delta2 += d3t.T * a2t

    delta1 /= m
    delta2 /= m

    # 添加正则的梯度
    delta1[:, 1:] = delta1[:, 1:] + (theta1[:, 1:] * lr) / m
    delta2[:, 1:] = delta2[:, 1:] + (theta2[:, 1:] * lr) / m

    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))

    return j, grad


if __name__ == '__main__':
    # 加载数据
    _data = loadmat('ex3data1.mat')
    _x = _data['X']
    _y = _data['y']
    print(_x.shape, _y.shape)

    # 对标签进行one-hot编码
    _encoder = OneHotEncoder(sparse=False)
    _y_one_hot = _encoder.fit_transform(_y)
    print(_y_one_hot.shape)
    print(_y[0], _y_one_hot[0, :])

    # 初始化设置
    _input_size = 400
    _hidden_size = 25
    _num_labels = 10
    _learning_rate = 1

    # 随机初始化完整网络参数大小的参数数组
    _params = (np.random.random(size=_hidden_size * (_input_size + 1) + _num_labels * (_hidden_size + 1)) - 0.5) * 0.25

    _m = _x.shape[0]

    # 将参数数组解开为每个层的参数矩阵
    _theta1 = np.matrix(np.reshape(_params[:_hidden_size * (_input_size + 1)], (_hidden_size, (_input_size + 1))))
    _theta2 = np.matrix(np.reshape(_params[_hidden_size * (_input_size + 1):], (_num_labels, (_hidden_size + 1))))

    print(_theta1.shape, _theta2.shape)

    # 测试 前向计算
    _a1, _z2, _a2, _z3, _h = forward(_x, _theta1, _theta2)
    print(_a1.shape, _z2.shape, _a2.shape, _z3.shape, _h.shape, _h[1, :])

    # 测试损失函数
    _j = cost(_params, _input_size, _hidden_size, _num_labels, _x, _y_one_hot, _learning_rate)
    print(_j)
    _j = cost_reg(_params, _input_size, _hidden_size, _num_labels, _x, _y_one_hot, _learning_rate)
    print(_j)

    # 测试 反向传播
    _j, _grad = back_propagation(_params, _input_size, _hidden_size, _num_labels, _x, _y_one_hot, _learning_rate)
    print(_j, _grad.shape)

    # 使用优化器寻找最优解
    _result = minimize(fun=back_propagation, x0=_params,
                       args=(_input_size, _hidden_size, _num_labels, _x, _y_one_hot, _learning_rate),
                       method='TNC', jac=True, options={'maxiter': 25})
    print(_result)

    # 拿到神经网络参数
    _theta1 = np.matrix(np.reshape(_result.x[:_hidden_size * (_input_size + 1)], (_hidden_size, (_input_size + 1))))
    _theta2 = np.matrix(np.reshape(_result.x[_hidden_size * (_input_size + 1):], (_num_labels, (_hidden_size + 1))))

    # 前向计算，做预测
    _a1, _z2, _a2, _z3, _h = forward(_x, _theta1, _theta2)
    _pre = np.array(np.argmax(_h, axis=1) + 1)
    print(_pre.shape)

    # 计算准确率
    correct = [1 if a == b else 0 for (a, b) in zip(_pre, _y)]
    accuracy = (sum(map(int, correct)) / float(len(correct)))
    print('accuracy = {0}%'.format(accuracy * 100))
