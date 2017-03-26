import os
import csv
import numpy as np
import kmeans

# Point to data directory here
# By default, we are pointing to '../data/'
data_dir = os.path.join('..','data')

# Load numeric data files into numpy arrays
X = np.genfromtxt(os.path.join(data_dir, 'kmeans_test_data.csv'), delimiter=',')

# TODO: Test update_assignments function, defined in kmeans.py

# TODO: Test update_centers function, defined in kmeans.py

# TODO: Test lloyd_iteration function, defined in kmeans.py

# TODO: Test kmeans_obj function, defined in kmeans.py

# TODO: Run experiments outlined in HW6 PDF

# For question 9 and 10
# from sklearn.decomposition import PCA
mnist_X = np.genfromtxt(os.path.join(data_dir, 'mnist_data.csv'), delimiter=',')