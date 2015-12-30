import sys
from os import path

sys.path.append(path.join(path.dirname(path.abspath(__file__)), '../..'))

import env

from mlkit.algo.decision_tree import tree
from mlkit.util import fileUtil
from mlkit.util import treePlotter

lensesTree = None
treeStorageFile = 'treeStorage.txt'
lensesFeatureNames = ['age', 'prescript', 'astigmatic', 'tearRate']


def getLensesTree():
    global lensesTree

    if lensesTree:
        pass
    elif path.exists(treeStorageFile):
        lensesTree = tree.loadTree(treeStorageFile)
    else:
        with open('lenses.txt') as fr:
            dataSet = [line.strip().split('\t') for line in fr.readlines() if line]
        lensesTree = tree.createTree(dataSet, lensesFeatureNames)
        tree.storeTree(lensesTree, treeStorageFile)

    return lensesTree


def classifyLenses(testVec):
    return getLensesTree().classify(lensesTree, lensesFeatureNames, testVec)


def printTree():
    print getLensesTree()


def plotTree():
    treePlotter.TreePlotter(getLensesTree()).createPlot()



if __name__ == '__main__':
    plotTree()