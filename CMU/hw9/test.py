import sys
import os
import numpy as np
import scipy.io
import cv2
import load
import pickle

# layer {
#   name: "data"
#   type: "Input"
#   top: "data"
#   input_param { shape: { dim: 1 dim: 3 dim: 32 dim: 32 } }
# }
sys.path.append("/home/haodong/caffe/python")
base_dir = ''

import caffe
from caffe.proto import caffe_pb2

PATH_PROTOTXT = base_dir+'1-nodata.prototxt'
PATH_MODEL = base_dir+'cifar3_iter_6000.caffemodel.h5'

file_dir = 'data_mat/test_data.mat'
data = scipy.io.loadmat(file_dir)
data = data['data']
images,images2 = load._convert_images(data)
count = 0
for item in images:
    print item.shape
    cv2.imwrite(str(count)+'.jpg',item)
    count += 1
    if count > 10:
        break


labels = data['labels']
data = data['data']
images,images2 = load._convert_images(data)

count = 0
net = caffe.Net(PATH_PROTOTXT, caffe.TEST, weights=PATH_MODEL)
print net.blobs['data'].data.shape
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
print net.blobs['data'].data.shape
#
# #
keys = [1,3,8]
# index = 0
# count = 0
# res_array = []
# error_num = 0
for item in images2:
#     cv2.imwrite(str(count)+'.jpg',item)
    item = item.reshape(1,3,32,32)
    print item.shape
    out = net.forward(item)
    print out['prob']
    break
    # index = np.argmax(out['prob'])
#     # type = -1
#     # if index == 1:
#     #     type = 0
#     # elif index == 3:
#     #     type = 1
#     # elif index == 8:
#     #     type = 2
#     # if type != labels.flatten()[count]:
#     #     error_num += 1
#     # count += 1
#
# # print len(res_array)
# print error_num
# print (float(np.sum(np.subtract(labels.flatten(),res_array))) / len(res_array))

