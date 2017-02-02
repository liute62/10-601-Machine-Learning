from data_utils import load_CIFAR10
import k_nearest_neighbor
import vis_utils
import numpy as np
import matplotlib.pyplot as plt

Xtr, Ytr, Xte, Yte = load_CIFAR10('../datasets/cifar-10-batches-py/') # a magic function we provide

print Xtr.shape
print Ytr.shape
print Ytr[0]
# flatten out all images to be one-dimensional
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3) # Xtr_rows becomes 50000 x 3072
Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3) # Xte_rows becomes 10000 x 3072

# Subsample the data for more efficient code execution in this exercise
num_training = 5000
mask = range(num_training)
X_train = Xtr_rows[mask]
y_train = Ytr[mask]
num_test = 500
mask = range(num_test)
X_test = Xte_rows[mask]
y_test = Yte[mask]

KNN = k_nearest_neighbor.KNearestNeighbor()

KNN.train(X_train,y_train)

# compute_distances_two_loops.
dists = KNN.compute_distances_two_loops(X_test)
print dists.shape

plt.imshow(dists, interpolation='none')

# Now implement the function predict_labels and run the code below:
# We use k = 1 (which is Nearest Neighbor).
y_test_pred = KNN.predict_labels(dists, k=1)

# Compute and print the fraction of correctly predicted examples
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print 'Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy)

