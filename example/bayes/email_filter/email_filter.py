import sys
from os import path

sys.path.append(path.join(path.dirname(path.abspath(__file__)), '../..'))

import env

from numpy import *

from mlkit.algo.bayes import bayes


def spamTest():
    docList=[]; classList = []; fullText =[]
    for i in range(1, 26):
        wordList = bayes.textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)

        wordList = bayes.textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    vocabList = bayes.createVocabList(docList)

    trainingSet = range(50); testSet=[]
    # create test set
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del trainingSet[randIndex]

    trainMat=[]; trainClasses = []
    # train the classifier (get probs) train
    for docIndex in trainingSet:
        trainMat.append(bayes.bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])

    pVectDict, pCateDict = bayes.train(array(trainMat), trainClasses)
    errorCount = 0
    # classify the remaining items
    for docIndex in testSet:
        wordVector = bayes.bagOfWords2VecMN(vocabList, docList[docIndex])
        if bayes.classify(array(wordVector), pVectDict, pCateDict) != classList[docIndex]:
            errorCount += 1
            print "classification error:", docList[docIndex]
    print 'the error rate is: ',float(errorCount) / len(testSet)


if __name__ == '__main__':
    spamTest()
