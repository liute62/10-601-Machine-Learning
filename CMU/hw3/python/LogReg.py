import os
import math
import numpy as np
import matplotlib.pyplot as plt


def LogReg_ReadInputs(filepath):
    
    #function that reads all four of the Logistic Regression csv files and outputs
    #them as such

    #Input
    #filepath : The path where all the four csv files are stored.
    XTrain = np.genfromtxt(os.path.join(filepath, 'LogReg_XTrain.csv'), delimiter=',')
    yTrain = np.genfromtxt(os.path.join(filepath, 'LogReg_yTrain.csv'), delimiter=',')
    XTest = np.genfromtxt(os.path.join(filepath, 'LogReg_XTest.csv'), delimiter=',')
    yTest = np.genfromtxt(os.path.join(filepath, 'LogReg_yTest.csv'), delimiter=',')
    #output 
    #XTrain : Nx(K+1) numpy matrix containing N number of K+1 dimensional training features
    #yTrain : Nx1 numpy vector containing the actual output for the training features
    #XTest  : nx(K+1) numpy matrix containing n number of K+1 dimensional testing features
    #yTest  : nx1 numpy vector containing the actual output for the testing features
    one_train = np.ones((XTrain.shape[0], 1))
    one_test = np.ones((XTest.shape[0], 1))
    XTrain = np.hstack((one_train, XTrain))
    XTest = np.hstack((one_test, XTest))
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
    
    y = y.reshape(y.shape[0], 1)
    y_s = np.dot(X, w)
    cll = 0.0
    for row in range(0, y_s.shape[0]):
        item = y_s[row, 0]
        if y[row] == 1:
            cll += math.log(sigmoid(item))
        else:
            cll += math.log(1 - sigmoid(item))
    return cll / X.shape[0]
    
def LogReg_CalcSG(x, y, w):
    
    #Function that calculates and returns the stochastic gradient value using a
    #particular data point (x, y).

    #Input
    #x : 1x(K+1) dimensional feature point
    #y : Actual output for the input x
    #w : weight vector 

    #Output
    #sg : gradient of the weight vector
    x_w = np.dot(x, w)
    loss = y - sigmoid(x_w)
    sg = (loss * x).transpose()
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
    w = np.subtract(w,eta * sg)
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
    mis_label = 0
    for row in range(0,X.shape[0]):
        item = X[row,:]
        item = item.reshape(1,X.shape[1])
        p = sigmoid(np.dot(item,w))
        if p > 0.5:
            yPred.append(1)
        else:
            yPred.append(0)
        if yPred[row] != y[row]:
            mis_label += 1
    perMiscl = mis_label * 1.0 / X.shape[0]
    return (yPred, perMiscl)

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
    w = np.zeros((XTrain.shape[1], 1))
    iter = 0
    for i in range(0, w.shape[0]):
        w[i, 0] = 0.5
    for epoch in range(0, 5):
        for sample in range(0, XTrain.shape[0]):
            x_row = XTrain[sample, :]
            x_row = x_row.reshape(1, x_row.shape[0])
            y_row = yTrain[sample]
            sg = LogReg_CalcSG(x_row, y_row, w)
            w = LogReg_UpdateParams(w, -sg, 0.5 / math.sqrt(iter + 1))
            iter += 1
            if iter % 200 == 0:
                pred_train,perMiscl_train = LogReg_PredictLabels(XTrain,yTrain,w)
                pred_test,perMiscl_test = LogReg_PredictLabels(XTest,yTest,w)
                trainPerMiscl.append(perMiscl_train)
                testPerMiscl.append(perMiscl_test)
    yPred = LogReg_PredictLabels(XTest,yTest,w)
    return (w, trainPerMiscl, testPerMiscl, yPred)

def sigmoid(x):
  return 1 / (1 + math.exp(-x))
    
def plot(train_error,test_error):
    #This function's results should be returned via gradescope and will not be evaluated in autolab.
    plt.plot(train_error)
    plt.title("training % of misclassified points vs SGA per 200 iterations")
    plt.xlabel('# / every 200 iterations')
    plt.ylabel('percentage of mis-classification train data')
    plt.show()
    plt.title("testing % of misclassified points vs SGA per 200 iterations")
    plt.plot(test_error)
    plt.xlabel('# / every 200 iterations')
    plt.ylabel('percentage of mis-classification test data')
    plt.show()
    return None
#
(XTrain, yTrain, XTest, yTest) = LogReg_ReadInputs('../data')
# #w = np.full((XTrain.shape[1],1),0.5)
# #pred_train,error_train = LogReg_PredictLabels(XTrain,yTrain,w)
# #cll = LogReg_CalcObj(XTrain,yTrain,np.ones((XTrain.shape[1],1)))
(w,train,test,y) = LogReg_SGA(XTrain,yTrain,XTest,yTest)
# # # print cll
plot(train,test)
