import scipy.io
import load
import const
import h5py
import numpy as np
import csv
import lmdb
import sys
sys.path.append("/home/haodong/caffe/examples/cifar3/workspace")
sys.path.append("/home/haodong/caffe/python")
from sklearn.decomposition import PCA
import caffe
import os

def load_train_data():

    '''
    Load cifar 3 Data set
    '''
    file_dir = const.DATA2_PATH
    print os.getcwd()
    data = scipy.io.loadmat(file_dir)
    labels = data['labels']
    data = data['data']
    print labels
    print data
    new_data = {}
    image_data = {}
    image, raw_image = load._convert_images(data)

    new_data['input'] = raw_image
    new_data['output'] = labels
    image_data['input'] = image
    image_data['output'] = labels

    return new_data, image_data

def load_test_data():

    '''
    Load cifar 3 Data set
    '''
    file_dir = const.TEST2_PATH
    data = scipy.io.loadmat(file_dir)
    data = data['data']

    new_data = {}
    image_data = {}
    image, raw_image = load._convert_images(data)

    new_data['input'] = raw_image
    image_data['input'] = image

    return new_data, image_data


def save_data_as_hdf5(hdf5_data_filename, data, isTest = False):
    '''
    HDF5 is one of the data formats Caffe accepts
    '''
    with h5py.File(hdf5_data_filename, 'w') as f:
        f['data'] = data['input'].astype(np.float32)
        if not isTest:
            f['label'] = data['output'].astype(np.float32)


def save_data_as_lmdb(lmdb_data_filename,X, isTest = False, isWhiten = False):
    data = X['input']
    if not isTest:
        label = X['output']
    map_size = data.nbytes * 10
    if isWhiten:
        float_data = np.zeros((data.shape[0],data.shape[1],data.shape[2],data.shape[3]))
    env = lmdb.open(lmdb_data_filename, map_size=map_size)
    with env.begin(write=True) as txn:
        # txn is a Transaction object
        for i in xrange(data.shape[0]):
            datum = caffe.proto.caffe_pb2.Datum()
            datum.channels = data.shape[1]
            datum.height = data.shape[2]
            datum.width = data.shape[3]
            if isWhiten:
                r_c = data[i][0].flatten()
                g_c = data[i][1].flatten()
                b_c = data[i][2].flatten()
                tmp_r = np.max(r_c) - np.min(r_c)
                tmp_g = np.max(g_c) - np.min(g_c)
                tmp_b = np.max(b_c) - np.min(b_c)
                float_data[i][0] = ((data[i][0] - np.min(r_c)) / float(tmp_r))
                float_data[i][1] = ((data[i][1] - np.min(g_c)) / float(tmp_g))
                float_data[i][2] = ((data[i][2] - np.min(b_c)) / float(tmp_b))
                datum.data = float_data[i].tobytes()  # or .tostring() if numpy < 1.9
            else:
                datum.data = data[i].tobytes()  # or .tostring() if numpy < 1.9
            if not isTest:
                datum.label = int(label[i])
            str_id = '{:08}'.format(i)
            # The encode is only essential in Python 3
            txn.put(str_id.encode('ascii'), datum.SerializeToString())


def compare(path_1,path_2,num):
    csv1 = csv.reader(open(path_1,'rU'), dialect=csv.excel_tab)

    csv2 = csv.reader(open(path_2,'rU'), dialect=csv.excel_tab)

    l1 = []
    l2 = []
    for line in csv1:
        l1.append(line)

    for line in csv2:
        l2.append(line)

    cnt = 0.0
    for i in range(0, num):
        if abs(float(l1[i][0]) - float(l2[i][0])) > 0.1:
            print i
            cnt = cnt + 1
    return 1 - cnt/num


def multiplicative_gaussian_noise(images, std=0.05):
    """
    Multiply with Gaussian noise.

    :param images: images (or data) in Caffe format (batch_size, height, width, channels)
    :type images: numpy.ndarray
    :param std: standard deviation of Gaussian
    :type std: float
    :return: images (or data) with multiplicative Gaussian noise
    :rtype: numpy.ndarray
    """

    assert images.ndim == 4
    assert images.dtype == np.float32

    return np.multiply(images, np.random.randn(images.shape[0], images.shape[1], images.shape[2],
                                                     images.shape[3]) * std + 1)


def additive_gaussian_noise(images, std=0.05):
    """
    Add Gaussian noise to the images.

    :param images: images (or data) in Caffe format (batch_size, height, width, channels)
    :type images: numpy.ndarray
    :param std: standard deviation of Gaussian
    :type std: float
    :return: images (or data) with additive Gaussian noise
    :rtype: numpy.ndarray
    """

    assert images.ndim == 4
    assert images.dtype == np.float32

    return images + np.random.randn(images.shape[0], images.shape[1], images.shape[2], images.shape[3]) * std