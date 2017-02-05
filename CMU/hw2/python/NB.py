import math
import numpy as np


# The logProd function takes a vector of numbers in logspace
# (i.e., x[i] = log p[i]) and returns the product of those numbers in logspace
# (i.e., logProd(x) = log(product_i p[i]))
def logProd(x):
    ## Inputs ##
    # x - 1D numpy ndarray
    ## Outputs ##
    # log_product - float
    product = 1
    for num in x:
        product *= num
    log_product = math.log(product)
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

    ## Outputs ##
    # D - (2 by V) numpy ndarray

    D = np.zeros([2, XTrain.shape[1]])
    return D


# The NB_YPrior function takes a set of training labels yTrain and
# returns the prior probability for class label 0
def NB_YPrior(yTrain):
    ## Inputs ##
    # yTrain - 1D numpy ndarray of length V

    ## Outputs ##
    # p - float

    p = 0
    return p


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


    yHat = np.ones(XTest.shape[0])
    return yHat


# The classificationError function takes two 1D arrays of class labels
# and returns the proportion of entries that disagree
def classificationError(yHat, yTruth):
    ## Inputs ##
    # yHat - 1D numpy ndarray of length m
    # yTruth - 1D numpy ndarray of length m

    ## Outputs ##
    # error - float

    error = 0
    return error
