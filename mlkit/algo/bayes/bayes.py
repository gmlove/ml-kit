'''
Created on Oct 19, 2010

@author: Peter
'''

from numpy import *


def createVocabList(dataSet):
    # create empty set
    vocabSet = set([])
    for document in dataSet:
        # union of the two sets
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print "the word: %s is not in vocabulary!" % word
    return returnVec


def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


def textParse(bigString):
    # input is big string, output is word list
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def train(trainMatrix, trainCategory):
    """Train the input data.

    Args:
        trainMatrix (ndarray):
        trainCategory (list):

    Returns:
        pVectDict (dict):
        pCateDict (dict):
    """
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    cateSet = set(trainCategory)

    pCate = dict([(cate, trainCategory.count(cate) / float(numTrainDocs)) for cate in cateSet])
    # initiate each occurrence to 1 to avoid '0 x anyInt = 0' problem
    pNum = dict([(cate, ones(numWords)) for cate in cateSet])
    # initiate denom to number of cateSet according to the pNum initiating
    pDenom = dict([(cate, float(len(cateSet))) for cate in cateSet])

    for i in range(numTrainDocs):
        cate = trainCategory[i]
        pNum[cate] += trainMatrix[i]
        pDenom[cate] += sum(trainMatrix[i])

    pVect = dict([(cate, log(pNum[cate] / pDenom[cate])) for cate in cateSet])

    return pVect, pCate


def classify(vec2Classify, pVectDict, pCateDict):
    """Using bayes algorithm to classify the target data.

    Args:
        vec2Classify (ndarray):
        pVectDict (dict):
        pCateDict (dict):

    Returns:
        class (any):
    """
    pTargetVec = [(cate, sum(vec2Classify * pVectDict[cate]) + log(pCateDict[cate])) for cate in pVectDict.keys()]
    return max(pTargetVec, key=lambda x: x[1])[0]


