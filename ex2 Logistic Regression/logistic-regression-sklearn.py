# coding= utf-8
"""
func: 逻辑回归，实现二分类，添加正则化（即重新写损失函数计算和梯度求导）
note: 坐标点和类别，使用sigmoid函数对输出的结果做二分类
other: 如何画出边界线？
"""

import numpy as np
import pandas as pd
from sklearn import linear_model  # 调用sklearn的线性回归包


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


if __name__ == '__main__':
    _file = 'ex2data2.txt'
    _data = pd.read_csv(_file, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
    _theta = np.zeros(3)

    # 查看数据
    show_data(_data, _theta, is_draw=False)

    _x, _y = dataset(_data)

    _x = np.matrix(_x.values)
    _y = np.matrix(_y.values)
    _theta = np.matrix(_theta)

    model = linear_model.LogisticRegression(penalty='l2', C=1.0)
    model.fit(_x, _y)

    acc = model.score(_x, _y)
    print(f'acc: {acc}')
