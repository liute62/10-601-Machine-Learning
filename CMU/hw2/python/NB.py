import numpy as np


# The logProd function takes a vector of numbers in logspace
# (i.e., x[i] = log p[i]) and returns the product of those numbers in logspace
# (i.e., logProd(x) = log(product_i p[i]))
def logProd(x):
    ## Inputs ##
    # x - 1D numpy ndarray
    ## Outputs ##
    # log_product - float
    log_product = np.sum(x)
    return log_product


# The NB_XGivenY function takes a training set XTrain and yTrain and
# Beta parameters alpha and beta, then returns a matrix containing
# MAP estimates of theta_yw for all words w and class labels y
def NB_XGivenY(XTrain, yTrain, alpha, beta):
    ## Inputs ##
    # XTrain - (n by V) numpy ndarray
    # yTrain - 1D numpy ndarray of length V
    # alpha - float
    # beta - float
    D = np.zeros([2, XTrain.shape[1]])
    total_onion = sum(yTrain)
    total_eco = yTrain.shape[0] - total_onion
    for words in range(0,XTrain.shape[1]):
        onion_num = 0
        econo_num = 0
        for docs in range(0,XTrain.shape[0]):

            if XTrain[docs][words] == 1:
                if yTrain[docs] == 1:
                    onion_num += 1
                else:
                    econo_num += 1
        D[0][words] = (econo_num + alpha - 1) / (total_eco + alpha + beta - 2)
        D[1][words] = (onion_num + alpha - 1) / (total_onion + alpha + beta - 2)

    # D - (2 by V) numpy ndarray
    return D


# The NB_YPrior function takes a set of training labels yTrain and
# returns the prior probability for class label 0
def NB_YPrior(yTrain):
    ## Inputs ##
    # yTrain - 1D numpy ndarray of length V

    ## Outputs ##
    # p - float
    total = np.sum(yTrain) * 1.0
    p = (yTrain.shape[0] - total) / yTrain.shape[0]
    return p * 1.0


# The NB_Classify function takes a matrix of MAP estimates for theta_yw,
# the prior probability for class 0, and uses these estimates to classify
# a test set.
def NB_Classify(D, p, XTest):
    ## Inputs ##
    # D - (2 by V) numpy ndarray
    # p - float
    # XTest - (m by V) numpy ndarray
    ## Outputs ##
    # yHat - 1D numpy ndarray of length m
    yHat = np.zeros(XTest.shape[0])
    for index in range(0,XTest.shape[0]):
       prod0 = np.ones(XTest.shape[1])
       prod1 = np.ones(XTest.shape[1])
       for words in range(0,XTest.shape[1]):
            if XTest[index][words] == 1:
                prod0[words] = D[0][words]
                prod1[words] = D[1][words]
            else:
                prod0[words] = 1 - D[0][words]
                prod1[words] = 1 - D[1][words]
       p0 = logProd(np.log(prod0)) + np.log(p)
       p1 = logProd(np.log(prod1)) + np.log(1-p)
       if p0 >= p1:
           yHat[index] = 0
       else:
           yHat[index] = 1
    return yHat


# The classificationError function takes two 1D arrays of class labels
# and returns the proportion of entries that disagree
def classificationError(yHat, yTruth):
    ## Inputs ##
    # yHat - 1D numpy ndarray of length m
    # yTruth - 1D numpy ndarray of length m
    ## Outputs ##
    # error - float
    total = yHat.shape[0]
    err = 0.0
    for index in range(1,total):
        if yHat[index] != yTruth[index]:
            err += 1
    error = err / total
    return error
