from numpy import *

def autoNorm(dataSet):
    """Automatically normalize dataSet.
    
    Args:
        dataSet (Matrix): input matrix
    
    Returns:
        normDataSet: normalized data
        ranges: ranges vector of every feature
        minVals: minimal values of every feature
    """
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)

    ranges = maxVals - minVals

    normDataSet = zeros(shape(dataSet))

    m = dataSet.shape[0]

    normDataSet = dataSet - tile(minVals, (m, 1))
    # element wise divide
    normDataSet = normDataSet / tile(ranges, (m, 1))

    return normDataSet, ranges, minVals
