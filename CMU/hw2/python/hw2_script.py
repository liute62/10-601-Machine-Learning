import os
import csv
import numpy as np
import NB
from scipy.stats import beta
import matplotlib.pyplot as plt

#Point to data directory here
#By default, we are pointing to '../data/'
alpha = 5
belta = 2
data_dir = os.path.join('..','data')

# Read vocabulary into a list
# You will not need the vocabulary for any of the homework questions.
# It is provided for your reference.
with open(os.path.join(data_dir, 'vocabulary.csv'), 'rb') as f:
    reader = csv.reader(f)
    vocabulary = list(x[0] for x in reader)

# TODO: Test logProd function, defined in NB.py

# Load numeric data files into numpy arrays
XTrain = np.genfromtxt(os.path.join(data_dir, 'XTrain.csv'), delimiter=',')
yTrain = np.genfromtxt(os.path.join(data_dir, 'yTrain.csv'), delimiter=',')
XTrainSmall = np.genfromtxt(os.path.join(data_dir, 'XTrainSmall.csv'), delimiter=',')
yTrainSmall = np.genfromtxt(os.path.join(data_dir, 'yTrainSmall.csv'), delimiter=',')
XTest = np.genfromtxt(os.path.join(data_dir, 'XTest.csv'), delimiter=',')
yTest = np.genfromtxt(os.path.join(data_dir, 'yTest.csv'), delimiter=',')

# TODO: Test NB_XGivenY function, defined in NB.py
D = NB.NB_XGivenY(XTrain,yTrain,alpha,belta)
# TODO: Test NB_YPrior function, defined in NB.py
p = NB.NB_YPrior(yTrain)
print yTrain.shape[0]
print p
# TODO: Test NB_Classify function, defined in NB.py
yHat = NB.NB_Classify(D,p,XTest)
# TODO: Test classificationError function, defined in NB.py
error = NB.classificationError(yHat,yTest)
print error
# TODO: Run experiments outlined in HW2 PDF
# https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.stats.beta.html
#
# fig, ax = plt.subplots(1, 1)
# mean, var, skew, kurt = beta.stats(alpha, belta, moments='mvsk')
# r = beta.rvs(alpha, belta, size=1000)
# ax.hist(r, normed=True, histtype='stepfilled', alpha=0.2)
# ax.legend(loc='best', frameon=False)
# plt.show()