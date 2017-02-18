import os
import math
import numpy as np


def LogReg_ReadInputs(filepath):
    
    #function that reads all four of the Logistic Regression csv files and outputs
    #them as such

    #Input
    #filepath : The path where all the four csv files are stored.

    #output 
    #XTrain : Nx(K+1) numpy matrix containing N number of K+1 dimensional training features
    #yTrain : Nx1 numpy vector containing the actual output for the training features
    #XTest  : nx(K+1) numpy matrix containing n number of K+1 dimensional testing features
    #yTest  : nx1 numpy vector containing the actual output for the testing features
    
    return (XTrain, yTrain, XTest, yTest)
    
def LogReg_CalcObj(X, y, w):
    
    #function that outputs the conditional log likelihood we want to maximize.

    #Input
    #w      : numpy weight vector of appropriate dimensions initialized to 0.5
    #AND EITHER
    #XTrain : Nx(K+1) numpy matrix containing N number of K+1 dimensional training features
    #yTrain : Nx1 numpy vector containing the actual output for the training features
    #OR
    #XTest  : nx(K+1) numpy matrix containing n number of K+1 dimensional testing features
    #yTest  : nx1 numpy vector containing the actual output for the testing features

    #Output
    #cll   : The conditional log likelihood we want to maximize
    
    cll =  0
    
    return cll
    
def LogReg_CalcSG(x, y, w):
    
    #Function that calculates and returns the stochastic gradient value using a
    #particular data point (x, y).

    #Input
    #x : 1x(K+1) dimensional feature point
    #y : Actual output for the input x
    #w : weight vector 

    #Output
    #sg : gradient of the weight vector
    
    sg = 0
    
    return sg
        
def LogReg_UpdateParams(w, sg, eta):
    
    #Function which takes in your weight vector w, the stochastic gradient
    #value sg and a learning constant eta and returns an updated weight vector w.

    #Input
    #w  : weight vector before update 
    #sg : gradient of the calculated weight vector using stochastic gradient ascent
    #eta: Learning rate

    #Output
    #w  : Updated weight vector
    
    return w
    
def LogReg_PredictLabels(X, y, w):
    
    #Function that returns the value of the predicted y along with the number of
    #errors between your predictions and the true yTest values

    #Input
    #w : weight vector 
    #AND EITHER
    #XTest : nx(K+1) numpy matrix containing m number of d dimensional testing features
    #yTest : nx1 numpy vector containing the actual output for the testing features
    #OR
    #XTrain : Nx(K+1) numpy matrix containing N number of K+1 dimensional training features
    #yTrain : Nx1 numpy vector containing the actual output for the training features
    
    #Output
    #yPred : An nx1 vector of the predicted labels for yTest/yTrain
    #perMiscl : The percentage of y's misclassified
    
    yPred = []
    perMiscl = 0
    
    return (yPred, PerMiscl)    

def LogReg_SGA(XTrain, yTrain, XTest, yTest):
    
    #Stochastic Gradient Ascent Algorithm function

    #Input
    #XTrain : Nx(K+1) numpy matrix containing N number of K+1 dimensional training features
    #yTrain : Nx1 numpy vector containing the actual output for the training features
    #XTest  : nx(K+1) numpy matrix containing n number of K+1 dimensional testing features
    #yTest  : nx1 numpy vector containing the actual output for the testing features

    #Output
    #w             : final weight vector
    #trainPerMiscl : a vector of percentages of misclassifications on your training data at every 200 gradient descent iterations
    #testPerMiscl  : a vector of percentages of misclassifications on your testing data at every 200 gradient descent iterations
    #yPred         : a vector of your predictions for yTest using your final w
    
    trainPerMiscl = []
    testPerMiscl = [] 
    
    return (w, trainPerMiscl, testPerMiscl, yPred)
    
def plot():     # This function's results should be returned via gradescope and will not be evaluated in autolab.
    
    return None