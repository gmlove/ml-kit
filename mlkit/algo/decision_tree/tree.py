'''
Created on Oct 12, 2010
Decision Tree Source Code for Machine Learning in Action Ch. 3

The decision tree construction algorithm is called ID3.

@author: Peter Harrington
'''

from math import log
import operator


def calcShannonEnt(dataSet):
    """Calculate Shannon entropy (a way to measure the disorder of the information)

    Args:
        dataSet (list): [feature-1, feature-2, ..., label]

    Returns:
        shannonEntropy: the Shannon entropy of the input data set
    """
    numEntries = len(dataSet)
    # the the number of unique elements and their occurrence
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts:
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)

    return shannonEnt


def reduceDataSet(dataSet, axis, value):
    """Create a new data set chopping out the axis according to the given value.

    Args:
        dataSet (list):
        axis (int):
        value (any):

    Returns:
        reduceDataSet (list): the chopped data set
    """
    reducedDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            # chop out axis used for splitting
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            reducedDataSet.append(reducedFeatVec)

    return reducedDataSet


def chooseBestFeatureToSplit(dataSet):
    """Choose best feature to split.

    Args:
        dataSet (list): [feature-1, feature-2, ..., label]

    Returns:
        bestFeatureIdx (int):
    """
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)

    bestInfoGain = 0.0; bestFeatureIdx = -1
    for i in range(numFeatures):
        # create a list of all the examples of this feature
        featList = [example[i] for example in dataSet]
        # get a set of unique values
        uniqueVals = set(featList)

        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = reduceDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)

        # calculate the info gain, ie reduction in entropy
        infoGain = baseEntropy - newEntropy

        # update the best infoGain if get better one
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeatureIdx = i

    # returns an integer
    return bestFeatureIdx


def majority(labelList):
    """Choose the most frequently appeared label.

    Args:
        labelList (list):

    Returns:
        label (any):
    """
    labelCount = {}
    for label in labelList:
        if label not in labelCount.keys():
            labelCount[label] = 0
        labelCount[label] += 1

    sortedLabelCount = sorted(labelCount.iteritems(), key=operator.itemgetter(1), reverse=True)

    majorityLabel = sortedLabelCount[0][0]
    return majorityLabel


def createTree(dataSet, featNames):
    """Create decision tree.
    The tree looks like: {'feature 1': {value1: {'feature 2': {...}}, value2: {'feature 3': {...}}}}

    Args:
        dataSet (list):
        featNames (list):

    Returns:
        tree (list):
    """
    labelList = [data[-1] for data in dataSet]
    if labelList.count(labelList[0]) == len(labelList):
        # stop splitting when all of the labels are equal
        return labelList[0]

    if len(dataSet[0]) == 1:
        # stop splitting when there are no more features in dataSet
        return majority(labelList)

    bestFeatIdx = chooseBestFeatureToSplit(dataSet)
    bestFeatName = featNames[bestFeatIdx]

    tree = {bestFeatName: {}}

    del featNames[bestFeatIdx]

    featValues = [data[bestFeatIdx] for data in dataSet]
    uniqueVals = set(featValues)
    for featValue in uniqueVals:
        # copy all of featNames, so trees don't mess up existing featNames
        subLabels = featNames[:]
        tree[bestFeatName][featValue] = createTree(reduceDataSet(dataSet, bestFeatIdx, featValue), subLabels)

    return tree


def classify(inputTree, featNames, testVec):
    """Using decision tree to classify the test data.

    Args:
        inputTree (dict):
        featNames (list):
        testVec (list):

    Returns:
        label (any):
    """
    featName = inputTree.keys()[0]
    nodeDict = inputTree[featName]

    featIndex = featNames.index(featName)

    featValue = testVec[featIndex]
    subTree = nodeDict[featValue]

    if isinstance(subTree, dict):
        label = classify(subTree, featNames, testVec)
    else:
        label = subTree

    return label


def storeTree(tree,filename):
    import pickle
    with open(filename,'w') as fw:
        pickle.dump(tree, fw)


def loadTree(filename):
    import pickle
    with open(filename) as fr:
        return pickle.load(fr)

