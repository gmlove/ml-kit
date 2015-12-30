import sys
from os import path

sys.path.append(path.join(path.dirname(path.abspath(__file__)), '../..'))

import env

from mlkit.algo.kNN import kNN
from mlkit.util import fileUtil
from mlkit.util import normalizer
import numpy


def datingClassTest():
    # hold out for test
    hoRatio = 0.50

    # load data set from file
    datingDataMat, datingLabels = fileUtil.file2matrix('datingTestSet2.txt')

    normMat, ranges, minVals = normalizer.autoNorm(datingDataMat)

    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)

    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = kNN.classify(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print "classifier result: %d, real answer: %d" % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0

    print "total count: %s, error count: %s" % (numTestVecs, errorCount)
    print "total error rate: %f" % (errorCount / float(numTestVecs))


def classifyPerson():
    strLabels = ['not at all', 'small doses', 'large doses']
    print ''

    ffMiles = float(raw_input('frequent flier miles earned per year: '))
    percentTats = float(raw_input('percentage of time spent playing video games: '))
    iceCream = float(raw_input('liters of ice cream consumed per year: '))

    datingDataMat, datingLabels = fileUtil.file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = normalizer.autoNorm(datingDataMat)

    testVec = numpy.array([ffMiles, percentTats, iceCream])
    classifierResult = kNN.classify((testVec - minVals) / ranges, normMat, datingLabels, 3)

    print ''
    print 'You will probably like this person:', strLabels[classifierResult - 1]


if __name__ == '__main__':
    # datingClassTest()
    classifyPerson()



