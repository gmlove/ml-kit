'''
Created on Oct 27, 2010
Logistic Regression Working Module

@author: Peter
'''
from numpy import *


def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))


def gradAscent(dataMatIn, classLabels):
    # convert to NumPy matrix
    dataMatrix = mat(dataMatIn)
    # convert to NumPy matrix
    labelMat = mat(classLabels).transpose()

    m, n = shape(dataMatrix)

    alpha = 0.001
    maxCycles = 500
    weights = ones((n, 1))
    # heavy on matrix operations
    for k in range(maxCycles):
        # matrix mult
        h = sigmoid(dataMatrix * weights)
        # vector subtraction
        error = labelMat - h
        # matrix mult
        weights = weights + alpha * dataMatrix.transpose() * error

    return weights


def stocGradAscent0(dataMatrix, classLabels):
    m, n = shape(dataMatrix)

    alpha = 0.01
    # initialize to all ones
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h

        weights = weights + alpha * error * dataMatrix[i]

    return weights


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = shape(dataMatrix)

    # initialize to all ones
    weights = ones(n)
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            # alpha decreases with iteration, does not go to 0 because of the constant
            alpha = 4 / (1.0 + j + i) + 0.0001

            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))

            error = classLabels[randIndex] - h

            weights = weights + alpha * error * dataMatrix[randIndex]

            del dataIndex[randIndex]

    return weights


def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    return 1.0 if prob > 0.5 else 0.0

