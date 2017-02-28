import os
import csv
import numpy as np
import perceptron
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
plt.show()

# TODO: Test perceptron_predict function, defined in perceptron.py

# TODO: Test perceptron_train function, defined in perceptron.py

# TODO: Test RBF_kernel function, defined in perceptron.py

# TODO: Test kernel_perceptron_predict function, defined in perceptron.py

# TODO: Test kernel_perceptron_train function, defined in perceptron.py

# TODO: Run experiments outlined in HW4 PDF