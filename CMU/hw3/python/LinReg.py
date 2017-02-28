import os
import math
import numpy as np
import matplotlib.pyplot as plt


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
    XTrain = np.genfromtxt(os.path.join(filepath, 'LinReg_XTrain.csv'), delimiter=',')
    yTrain = np.genfromtxt(os.path.join(filepath, 'LinReg_yTrain.csv'), delimiter=',')
    XTest = np.genfromtxt(os.path.join(filepath, 'LinReg_XTest.csv'), delimiter=',')
    yTest = np.genfromtxt(os.path.join(filepath, 'LinReg_yTest.csv'), delimiter=',')
    for index in range(0,XTrain.shape[1]):
        max_train = max(XTrain[0:XTrain.shape[0],index])
        max_test = max(XTest[0:XTest.shape[0],index])
        max_all = max(max_train,max_test)
        min_train = min(XTrain[0:XTrain.shape[0], index])
        min_test = min(XTest[0:XTest.shape[0], index])
        min_all = min(min_train, min_test)
        XTrain[:,index] = (XTrain[:,index] - min_all) / (max_all - min_all)
        XTest[:,index] = (XTest[:,index] - min_all) / (max_all - min_all)
    one_train = np.ones((XTrain.shape[0],1))
    one_test = np.ones((XTest.shape[0],1))
    XTrain = np.hstack((one_train,XTrain))
    XTest = np.hstack((one_test,XTest))
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
    # loss = np.zeros((X.shape[0],1))
    lossVal = 0
    y = y.reshape(y.shape[0],1)
    list = np.subtract(np.dot(X,w),y)
    sum = 0
    for row in range(0,list.shape[0]):
        item = list[row,0]
        sum += item * item
    lossVal = sum / X.shape[0]
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
    # x * w (1 * 14) (14 * 1) - y
    x_w = np.dot(x,w)
    loss = x_w - y
    sg = (loss * x).transpose() * 2
    # #print tmp
    # # (1 * 1)' * (1 * 14)
    # tmp = tmp.reshape(1,1)
    # x = x.reshape(1,x.shape[0])
    # tmp2 = np.dot(tmp,x)
    # # (1 * 14)
    # sg = np.sum(tmp2)
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
    w = np.subtract(w,eta * sg)
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
    w = np.zeros((XTrain.shape[1],1))
    iter = 0
    for i in range(0,w.shape[0]):
        w[i,0] = 0.5
    print w
    for epoch in range(0,100):
        #sg = np.zeros((XTrain.shape[1],1))
        for sample in range(0,XTrain.shape[0]):
            x_row = XTrain[sample,:]
            x_row = x_row.reshape(1,x_row.shape[0])
            y_row = yTrain[sample]
            sg = LinReg_CalcSG(x_row,y_row,w)
            w = LinReg_UpdateParams(w,sg,0.5 / math.sqrt(iter+1))
            iter += 1
        trainLoss.append(LinReg_CalcObj(XTrain,yTrain,w))
        testLoss.append(LinReg_CalcObj(XTest,yTest,w))
    return (w, trainLoss, testLoss)
    
def plot(trainLoss,testLoss,is_test):
    # This function's results should be returned via gradescope and will not be evaluated in autolab.
    plt.plot(trainLoss,'')
    plt.title('training losses versus the epoch.')
    plt.xlabel('#epoch times')
    plt.ylabel('trainLoss')
    plt.show()
    # plt.plot(testLoss)
    # plt.title('test losses versus the epoch.')
    # plt.xlabel('#epoch times')
    # plt.ylabel('testLoss')
    # plt.show()
    return None

#
(XTrain, yTrain, XTest, yTest) = LinReg_ReadInputs('../data')
# # # # lossVal = LinReg_CalcObj(XTrain,yTrain,np.ones((XTrain.shape[1],1)))
# # # # print lossVal
# # # #w = np.full((XTrain.shape[1],1),0.5)
# # # #sg = LinReg_CalcSG(XTrain[0,:],yTrain[0],w)
# # # #w2 = LinReg_UpdateParams(w,sg,0.1)
(w , trainLost, testLoss) = LinReg_SGD(XTrain, yTrain, XTest, yTest)
plot(trainLost,testLoss,1)