import sys
from os import path

sys.path.append(path.join(path.dirname(path.abspath(__file__)), '../..'))

import env

from os import listdir
from numpy import *

from mlkit.util import fileUtil
from mlkit.algo.kNN import kNN

def handwritingClassTest():
    hwLabels = []

    # load the training set
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)

    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        # take off .txt
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])

        hwLabels.append(classNumStr)

        trainingMat[i,:] = fileUtil.img2vector('trainingDigits/%s' % fileNameStr)

    # iterate through the test set
    testFileList = listdir('testDigits')
    errorCount = 0.0

    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        # take off .txt
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])

        vectorUnderTest = fileUtil.img2vector('testDigits/%s' % fileNameStr)
        classifierResult = kNN.classify(vectorUnderTest, trainingMat, hwLabels, 3)

        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)

        if (classifierResult != classNumStr):
            errorCount += 1.0

    print "\ntotal number of errors: %d" % errorCount
    print "\ntotal error rate: %f" % (errorCount/float(mTest))




if __name__ == '__main__':
    handwritingClassTest()



