'''
Created on Oct 14, 2010

@Usage:
    tree = {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
    treePlotter.TreePlotter(tree).createPlot()

@author: Peter Harrington
'''

import matplotlib.pyplot as plt


def getNumLeafs(tree):
    numLeafs = 0

    featName = tree.keys()[0]
    nodeDict = tree[featName]

    for key in nodeDict.keys():
        # test to see if the nodes are dictionaries, if not they are leaf nodes
        if type(nodeDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(nodeDict[key])
        else:
            numLeafs += 1

    return numLeafs


def getTreeDepth(tree):
    maxDepth = 0

    featName = tree.keys()[0]
    nodeDict = tree[featName]

    for key in nodeDict.keys():
        # test to see if the nodes are dictionaries, if not they are leaf nodes
        if type(nodeDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(nodeDict[key])
        else:
            thisDepth = 1
        maxDepth = max(thisDepth, maxDepth)

    return maxDepth


class TreePlotter:

    def __init__(self, tree):
        self.decisionNode = dict(boxstyle="sawtooth", fc="0.8")
        self.leafNode = dict(boxstyle="round4", fc="0.8")
        self.arrow_args = dict(arrowstyle="<-")
        self.plot_ax1 = None
        self.plot_xOff = None
        self.plot_yOff = None
        self.plot_totalW = None
        self.plot_totalD = None
        self.tree = tree



    def plotNode(self, nodeTxt, centerPt, parentPt, nodeType):
        self.plot_ax1.annotate(
            nodeTxt,
            xy=parentPt,
            xycoords='axes fraction',
            xytext=centerPt,
            textcoords='axes fraction',
            va="center",
            ha="center",
            bbox=nodeType,
            arrowprops=self.arrow_args
        )


    def plotMidText(self, cntrPt, parentPt, txtString):
        xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
        yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
        self.plot_ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


    def plotTree(self, tree, parentPt, nodeTxt):
        # the first key tells you what feat was split on

        # this determines the x width of this tree
        numLeafs = getNumLeafs(tree)
        depth = getTreeDepth(tree)

        # the text label for this node should be this
        featName = tree.keys()[0]

        cntrPt = (self.plot_xOff + (1.0 + float(numLeafs)) / 2.0 / self.plot_totalW, self.plot_yOff)

        self.plotMidText(cntrPt, parentPt, nodeTxt)
        self.plotNode(featName, cntrPt, parentPt, self.decisionNode)

        nodeDict = tree[featName]

        self.plot_yOff = self.plot_yOff - 1.0 / self.plot_totalD

        for key in nodeDict.keys():
            # test to see if the nodes are dictionaries, if not they are leaf nodes
            if type(nodeDict[key]).__name__ == 'dict':
                # recursive plot tree
                self.plotTree(nodeDict[key], cntrPt, str(key))
            # it's a leaf node print the leaf node
            else:
                self.plot_xOff = self.plot_xOff + 1.0 / self.plot_totalW

                self.plotNode(nodeDict[key], (self.plot_xOff, self.plot_yOff), cntrPt, self.leafNode)
                self.plotMidText((self.plot_xOff, self.plot_yOff), cntrPt, str(key))

        self.plot_yOff = self.plot_yOff + 1.0 / self.plot_totalD


    def createPlot(self):
        fig = plt.figure(1, facecolor='white')
        fig.clf()

        # no ticks
        axprops = dict(xticks=[], yticks=[])
        self.plot_ax1 = plt.subplot(111, frameon=False, **axprops)

        self.plot_totalW = float(getNumLeafs(self.tree))
        self.plot_totalD = float(getTreeDepth(self.tree))
        self.plot_xOff = -0.5 / self.plot_totalW;
        self.plot_yOff = 1.0;

        self.plotTree(self.tree, (0.5, 1.0), '')
        plt.show()

    def close(self):
        plt.close()


if __name__ == '__main__':
    listOfTrees =[
        {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
        {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
    ]
    TreePlotter(tree).createPlot()





