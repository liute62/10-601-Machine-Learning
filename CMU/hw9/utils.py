import scipy.io
import load
import const
import h5py
import numpy as np
import csv

def load_train_data():

    '''
    Load cifar 3 Data set
    '''
    file_dir = const.DATA_PATH
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
    file_dir = const.TEST_PATH
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


def compare(path_1,path_2,num):
    csv1 = csv.reader(open(path_1,'rU'), dialect=csv.excel_tab)

    csv2 = csv.reader(open(path_2,'rU'), dialect=csv.excel_tab)

    l1 = []
    l2 = []
    for line in csv1:
        l1.append(line)

    for line in csv2:
        l2.append(line)

    print l1
    print l2
    cnt = 0.0
    for i in range(0, num):
        if abs(float(l1[i][0]) - float(l2[i][0])) > 0.1:
            cnt = cnt + 1
    return 1 - cnt/num