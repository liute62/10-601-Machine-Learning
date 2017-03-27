import numpy as np
import math

########################################################################
#######  you should maintain the  return type in starter codes   #######
########################################################################

def update_assignments(X, C):
  # Input:
  #   X is the data matrix (n, d),  2-d array
  #   C is the cluster centers (k, d), 2-d array
  # Output:
  #   a is the cluster assignments (n,), 1-d array
  a = np.zeros(X.shape[0])
  for i in range(0,X.shape[0]):
    point = X[i, :]
    min_index = 0
    min_dis = float('inf')
    for j in range(0,C.shape[0]):
      center = C[j,:]
      dis = sum((point - center) ** 2)
      if dis < min_dis:
        min_index = j
        min_dis = dis
    a[i] = min_index
  return a

def update_centers(X, C, a):
  # Input:
  #   X is the data matrix (n, d),  2-d array
  #   C is the cluster centers (k, d), 2-d array
  #   a is the cluster assignments (n,), 1-d array
  # Output:
  #   C is the new cluster centers (k, d), 2-d array
  for i in range(0,C.shape[0]):
    array = []
    for j in range(0,a.shape[0]):
      if a[j] == i:
        point = X[j,:]
        array.append(point)
    if len(array) == 0:
      continue
    sum_array = (0,0)
    for k in range(0,len(array)):
      sum_array += array[k]
    sum_array = sum_array / len(array)
    C[i] = sum_array
    # min_dis = float('inf')
    # ctr = array[0]
    # for k in range(0,len(array)):
    #   dis = 0
    #   for m in range(0,len(array)):
    #     if k != m:
    #       dis += sum((array[k] - array[m])**2)
    #   if dis < min_dis:
    #      min_dis = dis
    #      ctr = array[k]
    # C[i] = ctr
  return C



def lloyd_iteration(X, C):
  # Input:
  #   X is the data matrix (n, d),  2-d array
  #   C is the initial cluster centers (k, d), 2-d array
  # Output:
  #   C is the cluster centers (k, d), 2-d array
  #   a is the cluster assignments (n,), 1-d array
  a = np.zeros(X.shape[0])
  a = update_assignments(X, C)
  last_obj_loss = 0
  is_converge = False
  while not is_converge:
    obj_loss = kmeans_obj(X,C,a)
    if last_obj_loss != obj_loss:
      update_centers(X,C,a)
      a = update_assignments(X,C)
      last_obj_loss = obj_loss
    else:
      is_converge = True
  return (C, a)

def kmeans_obj(X, C, a):
  # Input:
  #   X is the data matrix (n, d),  2-d array
  #   C is the cluster centers (k, d), 2-d array
  #   a is the cluster assignments (n,), 1-d array
  # Output:
  #   obj is the k-means objective of the provided clustering, scalar, float
  obj = 0.0
  for i in range(0,X.shape[0]):
    point = X[i,:]
    assigned = a[i]
    ctr_point = C[int(assigned),:]
    dis = sum((point - ctr_point) ** 2)
    obj += dis
  return obj


########################################################################
#######          DO NOT MODIFY, BUT YOU SHOULD UNDERSTAND        #######
########################################################################

# kmeans_cluster will be used in the experiments, it is available after you 
# have implemented lloyd_iteration and kmeans_obj.

def kmeans_cluster(X, k, init, num_restarts):
  n = X.shape[0]
  # Variables for keeping track of the best clustering so far
  best_C = None
  best_a = None
  best_obj = np.inf
  for i in range(num_restarts):
    if init == "random":
      perm = np.random.permutation(range(n))
      C = np.copy(X[perm[0:k]])
    elif init == "kmeans++":
      C = kmpp_init(X, k)
    elif init == "fixed":
      C = np.copy(X[0:k])
    else:
      print "No such module"
    # Run the Lloyd iteration until convergence
    (C, a) = lloyd_iteration(X, C)
    # Compute the objective value
    obj = kmeans_obj(X, C, a)
    if obj < best_obj:
      best_C = C
      best_a = a
      best_obj = obj
  return (best_C, best_a, best_obj)



########################################################################
#######     DO NOT MODIFY, READ ONLY IF YOU ARE INTERESTED       #######
########################################################################

def kmpp_init(X, k):
  n = X.shape[0]
  sq_distances = np.ones(n)
  center_ixs = list()
  for j in range(k):
    # Choose a new center index using D^2 weighting
    ix = discrete_sample(sq_distances)
    # Update the squared distances for all points
    deltas = X - X[ix]
    for i in range(n):
      sq_dist_to_ix = np.power(np.linalg.norm(deltas[i], 2), 2)
      sq_distances[i] = min(sq_distances[i], sq_dist_to_ix)
    # Append this center to the list of centers
    center_ixs.append(ix)
  # Output the chosen centers
  C = X[center_ixs]
  return np.copy(C)


def discrete_sample(weights):
  total = np.sum(weights)
  t = np.random.rand() * total
  p = 0.0
  for i in range(len(weights)):
    p = p + weights[i];
    if p > t:
      ix = i
      break
  return ix