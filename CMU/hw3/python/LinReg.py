import os
import math
import numpy as np


def LinReg_ReadInputs(filepath):
    
    #function that reads all four of the Linear Regression csv files and outputs
    #them as such

    #Input
    #filepath : The path where all the four csv files are stored.

    #output 
    #XTrain : NxK+1 numpy matrix containing N number of K+1 dimensional training features
    #yTrain : Nx1 numpy vector containing the actual output for the training features
    #XTest  : nxK+1 numpy matrix containing n number of K+1 dimensional testing features
    #yTest  : nx1 numpy vector containing the actual output for the testing features
    
    
    return (XTrain, yTrain, XTest, yTest)
    
def LinReg_CalcObj(X, y, w):
    
    #function that outputs the value of the loss function L(w) we want to minimize.

    #Input
    #w      : numpy weight vector of appropriate dimensions
    #AND EITHER
    #XTrain : Nx(K+1) numpy matrix containing N number of K+1 dimensional training features
    #yTrain : Nx1 numpy vector containing the actual output for the training features
    #OR
    #XTest  : nx(K+1) numpy matrix containing n number of K+1 dimensional testing features
    #yTest  : nx1 numpy vector containing the actual output for the testing features
    
    #Output
    #loss   : The value of the loss function we want to minimize
    
    lossVal = 0
    
    return lossVal



def LinReg_CalcSG(x, y, w):
    
    #Function that calculates and returns the stochastic gradient value using a
    #particular data point (x, y).

    #Input
    #x : 1x(K+1) dimensional feature point
    #y : Actual output for the input x
    #w : (K+1)x1 dimensional weight vector 

    #Output
    #sg : gradient of the weight vector
    
    sg = 0
    
    return sg

def LinReg_UpdateParams(w, sg, eta):
    
    #Function which takes in your weight vector w, the stochastic gradient
    #value sg and a learning constant eta and returns an updated weight vector w.

    #Input
    #w  : (K+1)x1 dimensional weight vector before update 
    #sg : gradient of the calculated weight vector using stochastic gradient descent
    #eta: Learning rate

    #Output
    #w  : Updated weight vector
    
    return w
    
def LinReg_SGD(XTrain, yTrain, XTest, yTest):
    
    #Stochastic Gradient Descent Algorithm function

    #Input
    #XTrain : Nx(K+1) numpy matrix containing N number of K+1 dimensional training features
    #yTrain : Nx1 numpy vector containing the actual output for the training features
    #XTest  : nx(K+1) numpy matrix containing n number of K+1 dimensional test features
    #yTest  : nx1 numpy vector containing the actual output for the test features
    
    #Output
    #w    : Updated Weight vector after completing the stochastic gradient descent
    #trainLoss : vector of training loss values at each epoch
    #testLoss : vector of test loss values at each epoch
    
    trainLoss = []
    testLoss = []
    
    return (w, trainLoss, testLoss)
    
def plot():     # This function's results should be returned via gradescope and will not be evaluated in autolab.
    
    return None
    

    
    