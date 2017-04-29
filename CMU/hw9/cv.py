import cv2
import sklearn
import sys
import scipy
import load
from sklearn.svm import SVC
import scipy.io

base_dir = '/home/haodong/caffe/examples/cifar10/'
sys.path.append("/home/haodong/caffe/python")

file_dir = 'data_mat/data_batch.mat'
data = scipy.io.loadmat(file_dir)
labels = data['labels']
data = data['data']
images = load._convert_images(data)

clf = SVC()
clf.fit(data, labels)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)