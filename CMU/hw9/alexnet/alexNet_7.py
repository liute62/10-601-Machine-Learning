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
        #train_data, __ = utils.load_train_data()
        #train_data['input'] = (np.array(train_data['input'], dtype=np.float32))
        #train_data['input'] = utils.additive_gaussian_noise(train_data['input'])
        # utils.save_data_as_lmdb('cifar7_train_data_lmdb',train_data)
        caffe.set_mode_gpu()
        solver = caffe.get_solver('alexnet_solver_7.prototxt')
        solver.solve()
        pass

    def test(self,str):
        test_data, __ = utils.load_test_data()
        # utils.save_data_as_lmdb('cifar5_test_data_lmdb', test_data, True)
        result = self.__get_predicted_output('alexnet_result_7.prototxt', 'cifar3_7_iter_'+str+'.caffemodel.h5')
        res = np.zeros(len(result), dtype=int)
        for i in xrange(len(result)):
            res[i] = (np.argmax(result[i]))
        np.savetxt("results7.csv", res.astype(dtype=int))

alex = AlexNet()
alex.test('170000')