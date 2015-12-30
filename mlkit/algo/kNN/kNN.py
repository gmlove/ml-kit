'''
Created on Sep 16, 2010
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (MxN)
            labels: data set labels (Mx1)
            k: number of neighbors to use for comparison (should be an odd number)

Output:     the most popular class label

@author: pbharrin
'''
from numpy import *
import operator
from os import listdir

def classify(inX, dataSet, labels, k):

    # calculate distance from inX to each known vector

    # count of known vectors
    dataSetSize = dataSet.shape[0]
    # replicate input vector to a MxN matrix, and minus known vectors
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    # square each point
    sqDiffMat = diffMat ** 2
    # sum according to row, result in a Mx1 vector
    sqDistances = sqDiffMat.sum(axis=1)
    # sqrt each point
    distances = sqDistances ** 0.5

    # find the nearest k vectors of the known vectors, calculate label frequency

    # sort each point, return the index value vector of the sorted vector
    sortedDistIndicies = distances.argsort()
    # calculate the most frequent labels of the k vectors
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)

    # return the most frequent label
    return sortedClassCount[0][0]

