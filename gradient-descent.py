#coding=utf-8

import numpy as np
import random
from numpy import genfromtxt


def getData(dataSet):
    m, n = np.shape(dataSet)
    trainData = np.ones((m, n))
    trainData[:,:-1] = dataSet[:,:-1]
    trainLabel = dataSet[:,-1]
    return trainData, trainLabel


# 批量梯度下降
def batchGradientDescent(x, y, theta, alpha, m, maxIterations):
    xTrains = x.transpose()
    for i in range(0, maxIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        # print loss
        gradient = np.dot(xTrains, loss) / m
        theta = theta - alpha * gradient
    return theta


def predict(x, theta):
    m, n = np.shape(x)
    xTest = np.ones((m, n+1))
    xTest[:, :-1] = x
    yP = np.dot(xTest, theta)
    return yP


# 格式[[x[0], x[1], y],
#      [x[0], x[1], y]....]
dataSet = np.array([[1.1, 1.5, 2.5],
                    [1.3, 1.9, 3.2],
                    [1.5, 2.3, 3.9],
                    [1.7, 2.7, 4.6],
                    [1.9, 3.1, 5.3],
                    [2.1, 3.5, 6],
                    [2.3, 3.9, 6.7],
                    [2.5, 4.3, 7.4],
                    [2.7, 4.7, 8.1],
                    [2.9, 5.1, 8.8]])
trainData, trainLabel = getData(dataSet)
m, n = np.shape(trainData)
theta = np.ones(n)
alpha = 0.1
maxIteration = 5000
theta = batchGradientDescent(trainData, trainLabel, theta, alpha, m, maxIteration)
x = np.array([[3.1, 5.5], [3.3, 5.9], [3.5, 6.3], [3.7, 6.7], [3.9, 7.1]])
print(predict(x, theta))
print(theta)

# theta[0]*x0 + theta[1]*x1 +theta[2] = y
