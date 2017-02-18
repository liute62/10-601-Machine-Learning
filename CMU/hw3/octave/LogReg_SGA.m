%Stochastic Gradient Ascent Algorithm function

%Input
%XTrain : NxK+1 matrix containing N number of K+1 dimensional training features
%yTrain : Nx1 vector containing the actual output for the training features
%XTest  : nxK+1 matrix containing n number of K+1 dimensional testing features
%yTest  : nx1 vector containing the actual output for the testing features


%Output
%w             : final weight vector
%trainPerMiscl : a vector of percentages of misclassifications on your training data at every 200 gradient descent iterations
%testPerMiscl  : a vector of percentages of misclassifications on your testing data at every 200 gradient descent iterations
%yPred         : a vector of your predictions for yTest using your final w
   
function [w, trainPerMiscl, testPerMiscl, yPred] = LogReg_SGA (XTrain, yTrain, XTest, yTest)

endfunction
