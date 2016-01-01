import sys
from os import path

sys.path.append(path.join(path.dirname(path.abspath(__file__)), '../..'))

import env

from numpy import *
import feedparser
import operator


from mlkit.algo.bayes import bayes


def calcMostFreq(vocabList, fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]


def localWords(feed1, feed0):
    docList=[]; classList = []; fullText =[]
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        wordList = bayes.textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1) #NY is class 1
        wordList = bayes.textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    vocabList = bayes.createVocabList(docList)
    top30Words = calcMostFreq(vocabList, fullText)

    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])

    trainingSet = range(2 * minLen); testSet=[]
    # create test set
    for i in range(20):
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

    print 'the error rate is: ', float(errorCount) / len(testSet)

    return vocabList, pVectDict


def getTopWords(ny, sf):
    vocabList, pVectDict = localWords(ny, sf)
    p0V = pVectDict[0]
    p1V = pVectDict[1]

    topNY=[]; topSF=[]
    for i in range(len(p0V)):
        if p0V[i] > -6.0 : topSF.append((vocabList[i],p0V[i]))
        if p1V[i] > -6.0 : topNY.append((vocabList[i],p1V[i]))

    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print "SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**"
    for item in sortedSF[:20]:
        print item[0]

    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print "NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**"
    for item in sortedNY[:20]:
        print item[0]


if __name__ == '__main__':
    ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
    getTopWords(ny, sf)
