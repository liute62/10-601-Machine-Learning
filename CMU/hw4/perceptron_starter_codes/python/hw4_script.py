import os
import csv
import numpy as np
import perceptron as pe
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# Point to data directory here
# By default, we are pointing to '../data/'
data_dir = os.path.join('..','data')

# Load numeric data files into numpy arrays
XTrain = np.genfromtxt(os.path.join(data_dir, 'XTrain.csv'), delimiter=',')
yTrain = np.genfromtxt(os.path.join(data_dir, 'yTrain.csv'), delimiter=',')
XTest = np.genfromtxt(os.path.join(data_dir, 'XTest.csv'), delimiter=',')
yTest = np.genfromtxt(os.path.join(data_dir, 'yTest.csv'), delimiter=',')

# Visualize the image
idx = 0
datapoint = XTrain[idx, 1:]
plt.imshow(datapoint.reshape((28,28), order = 'F'), cmap='gray')
#plt.show()

# TODO: Test perceptron_predict function, defined in perceptron.py

# TODO: Test perceptron_train function, defined in perceptron.py

# TODO: Test RBF_kernel function, defined in perceptron.py

# TODO: Test kernel_perceptron_predict function, defined in perceptron.py

# TODO: Test kernel_perceptron_train function, defined in perceptron.py
# TODO: Run experiments outlined in HW4 PDF

#1
# a0 = np.zeros(XTrain.shape[1])
# a1 = pe.perceptron_train(a0,XTrain,yTrain,10)
# error_num = 0
# for i in range(0,XTest.shape[0]):
#     y_pre = pe.perceptron_predict(a1,XTest[i])
#     y_real = yTest[i]
#     if y_real != y_pre:
#         error_num += 1
# error_rate = error_num * 1.0 / XTest.shape[0]
# print error_rate

#2
for sigma in (0.01,0.1,1,10,100,1000):
    a0 = np.zeros(XTrain.shape[0])
    a1 = pe.kernel_perceptron_train(a0,XTrain,yTrain,2,sigma)
    error_num = 0;
    for i in range(0, XTest.shape[0]):
        y1 = pe.kernel_perceptron_predict(a1,XTrain,yTrain,XTest[i],sigma)
        y2 = yTest[i]
        if y1 != y2:
            error_num += 1
    error_rate = error_num * 1.0 / XTest.shape[0]
    print sigma,error_rate

#3
