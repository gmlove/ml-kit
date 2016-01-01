import unittest
from numpy import *

from mlkit.algo.bayes import bayes

class TestBayes(unittest.TestCase):

    def loadDataSet(self):
        postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                     ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                     ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                     ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                     ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                     ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
        # 1 is abusive, 0 not
        classVec = [0, 1, 0, 1, 0, 1]
        return postingList,classVec

    def test_classify_by_set_of_model(self):
        listOPosts, listClasses = self.loadDataSet()
        myVocabList = bayes.createVocabList(listOPosts)

        trainMat=[]
        for postinDoc in listOPosts:
            trainMat.append(bayes.setOfWords2Vec(myVocabList, postinDoc))
        pVectDict, pCateDict = bayes.train(array(trainMat), listClasses)
        print pVectDict
        print pCateDict

        testEntry = ['love', 'my', 'dalmation']
        thisDoc = array(bayes.setOfWords2Vec(myVocabList, testEntry))
        print testEntry, 'classified as: ', bayes.classify(thisDoc, pVectDict, pCateDict)

        testEntry = ['stupid', 'garbage']
        thisDoc = array(bayes.setOfWords2Vec(myVocabList, testEntry))
        print testEntry, 'classified as: ', bayes.classify(thisDoc, pVectDict, pCateDict)

    def test_classify_by_bag_of_model(self):
        listOPosts, listClasses = self.loadDataSet()
        myVocabList = bayes.createVocabList(listOPosts)

        trainMat=[]
        for postinDoc in listOPosts:
            trainMat.append(bayes.bagOfWords2VecMN(myVocabList, postinDoc))
        pVectDict, pCateDict = bayes.train(array(trainMat), listClasses)
        print pVectDict
        print pCateDict

        testEntry = ['love', 'my', 'dalmation']
        thisDoc = array(bayes.bagOfWords2VecMN(myVocabList, testEntry))
        print testEntry, 'classified as: ', bayes.classify(thisDoc, pVectDict, pCateDict)

        testEntry = ['stupid', 'garbage']
        thisDoc = array(bayes.bagOfWords2VecMN(myVocabList, testEntry))
        print testEntry, 'classified as: ', bayes.classify(thisDoc, pVectDict, pCateDict)


if __name__ == '__main__':
    unittest.main(verbosity=2)
