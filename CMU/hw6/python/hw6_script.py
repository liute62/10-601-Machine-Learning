import os
import csv
import numpy as np
import kmeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Point to data directory here
# By default, we are pointing to '../data/'
data_dir = os.path.join('..','data')

# Load numeric data files into numpy arrays
X = np.genfromtxt(os.path.join(data_dir, 'kmeans_test_data.csv'), delimiter=',')
print X.shape[0],X.shape[1]
print X[0,0]
# TODO: Test update_assignments function, defined in kmeans.py

# TODO: Test update_centers function, defined in kmeans.py

# TODO: Test lloyd_iteration function, defined in kmeans.py

# TODO: Test kmeans_obj function, defined in kmeans.py

# TODO: Run experiments outlined in HW6 PDF
# objs = []
# last_obj = 0
# for i in range(1,12):
#     c,a,obj = kmeans.kmeans_cluster(X,i,'random',10)
#     objs.append(obj)
#     print i,obj
# for i in range(0,len(objs)-3):
#     a = objs[i] - objs[i+1]
#     b = objs[i+2] - objs[i+3]
#     print b - a
# plt.plot(objs,label="point")
# plt.ylabel('loss value')
# plt.show()
# # For question 9 and 10
# # from sklearn.decomposition import PCA
# mnist_X = np.genfromtxt(os.path.join(data_dir, 'mnist_data.csv'), delimiter=',')
# X = X[0:7]
# X[0][0] = 1.0
# X[0][1] = 1.0
# X[1][0] = 1.5
# X[1][1] = 2.0
# X[2][0] = 3.0
# X[2][1] = 4.0
# X[3][0] = 5.0
# X[3][1] = 7.0
# X[4][0] = 3.5
# X[4][1] = 5.0
# X[5][0] = 4.5
# X[5][1] = 5.0
# X[6][0] = 3.5
# X[6][1] = 4.5
# x = []
# y = []
# for item in X:
#     x.append(item[0])
#     y.append(item[1])
# plt.scatter(x,y)
# plt.show()
# obj = 0
# for i in range(0,1000):
#     (best_C, best_a, best_obj) = kmeans.kmeans_cluster(X,9,'kmeans++',1)
#     obj += best_obj
#     # print best_obj
# print obj / 1000

X = np.genfromtxt(os.path.join(data_dir, 'mnist_data.csv'), delimiter=',')
print X.shape[0],X.shape[1]
print X[0,0]
#
x_reduced = PCA(n_components=5).fit_transform(X)
print x_reduced.shape[0],x_reduced.shape[1]


(best_C, best_a, best_obj) = kmeans.kmeans_cluster(x_reduced, 3, 'fixed', 1)
print best_obj
print best_C
print best_a