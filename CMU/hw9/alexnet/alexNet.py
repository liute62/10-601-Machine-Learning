import sys
sys.path.append("/home/haodong/caffe/examples/cifar3/workspace")
import utils
import const
sys.path.append("/home/haodong/caffe/python")
import caffe
import copy
import numpy as np
import matplotlib.pyplot as plt


class AlexNet:

    def __init__(self):
        pass

    def __get_predicted_output(self,deploy_prototxt_filename, caffemodel_filename, net=None):
        '''
        Get the predicted output, i.e. perform a forward pass
        '''
        if net is None:
            net = caffe.Net(deploy_prototxt_filename, caffemodel_filename, caffe.TEST)
        out = net.forward()
        print('out: {0}'.format(out))
        return out[net.outputs[0]]

    def train(self):
        train_data, __ = utils.load_train_data()
        print train_data['input'].shape
        print train_data['output'].shape
        utils.save_data_as_hdf5(const.HDF5_TRAIN_DATA_PATH,train_data)
        utils.save_data_as_hdf5(const.HDF5_TEST_DATA_PATH,train_data)
        caffe.set_mode_gpu()
        solver = caffe.get_solver(const.ALEXNET_SOLVER)
        solver.solve()
        pass

    def test(self):
        test_data, __ = utils.load_test_data()
        utils.save_data_as_hdf5(const.HDF5_RESULT_DATA_PATH, test_data, True)
        result = self.__get_predicted_output(const.ALEXNET_RESULT, 'alexnet/cifar3_1_iter_60000.caffemodel.h5')
        res = np.zeros(len(result), dtype=int)
        for i in xrange(len(result)):
            res[i] = (np.argmax(result[i]))
            # print res[i]
        # print len(res)
        np.savetxt("alexnet/result.csv", res.astype(dtype=int))

alex = AlexNet()
alex.train()
alex.test()
print utils.compare('label.csv','alexnet/result.csv',3000)