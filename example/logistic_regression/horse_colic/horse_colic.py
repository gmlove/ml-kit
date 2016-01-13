import sys
from os import path

sys.path.append(path.join(path.dirname(path.abspath(__file__)), '../..'))

import env

from mlkit.algo.logistic_regression import logistic_regression
from numpy import *



def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')

    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')

        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))

        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))

    trainWeights = logistic_regression.stocGradAscent1(array(trainingSet), trainingLabels, 1000)

    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0

        currLine = line.strip().split('\t')

        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))

        if int(logistic_regression.classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1

    errorRate = (float(errorCount) / numTestVec)
    print "the error rate of this test is: %f" % errorRate

    return errorRate


def multiTest():
    numTests = 10; errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()

    print "after %d iterations the average error rate is: %f" % (numTests, errorSum / float(numTests))



if __name__ == '__main__':
    # colicTest()
    multiTest()
