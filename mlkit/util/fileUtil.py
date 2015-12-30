
from numpy import *

def _line2vector(line, seperator):
    line = line.strip()
    return line.split(seperator)


def file2matrix(filename, seperator='\t'):
    numberOfLines, featureCount = 0, 0
    with open(filename) as fr:
        # get the number of lines in the file
        lines = fr.readlines()
        numberOfLines = len(lines)
        # get feature count
        if numberOfLines: featureCount = len(_line2vector(lines[0], seperator)) - 1

    # prepare matrix to return
    featureMatrix = zeros((numberOfLines, featureCount))
    # prepare labels to return
    classLabelVector = []

    with open(filename) as fr:
        index = 0
        for line in fr.readlines():
            listFromLine = _line2vector(line, seperator)
            # each value will be automatically cast to float64
            featureMatrix[index, :] = listFromLine[0:-1]
            classLabelVector.append(int(listFromLine[-1]))
            index += 1

    return featureMatrix, classLabelVector


def img2vector(filename):
    imgVector = zeros((1, 1024))
    with open(filename) as fr:
        for i in range(32):
            lineStr = fr.readline()
            for j in range(32):
                imgVector[0, 32 * i + j] = int(lineStr[j])
    return imgVector
