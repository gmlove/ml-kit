import unittest
from numpy import *

from mlkit.algo.logistic_regression import logistic_regression

def loadDataSet():
    dataMat = [];
    labelMat = []
    fr = open('logistic_regression_testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def plotBestFit(weights):
    import matplotlib.pyplot as plt

    dataMat, labelMat = loadDataSet()

    dataArr = array(dataMat)
    n = shape(dataArr)[0]

    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1]); ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1]); ycord2.append(dataArr[i, 2])

    fig = plt.figure()

    ax = fig.add_subplot(111)

    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')

    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    print weights

    ax.plot(x, y)

    plt.xlabel('X1'); plt.ylabel('X2');

    plt.show()


if __name__ == '__main__':
    dataMat, labelMat = loadDataSet()
    weights = logistic_regression.gradAscent(dataMat, labelMat)
    plotBestFit(weights.getA())

