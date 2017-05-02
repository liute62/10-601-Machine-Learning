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
        # train_data['input'] = train_data['input'] / 255.0
        utils.save_data_as_lmdb('cifar5_train_data_lmdb',train_data)
        caffe.set_mode_gpu()
        solver = caffe.get_solver('alexnet_solver_5.prototxt')
        solver.solve()
        pass

    def test(self):
        test_data, __ = utils.load_test_data()
        # utils.save_data_as_lmdb('cifar5_test_data_lmdb', test_data, True)
        result = self.__get_predicted_output('alexnet_result_5.prototxt', 'cifar3_5_iter_200000.caffemodel.h5')
        res = np.zeros(len(result), dtype=int)
        for i in xrange(len(result)):
            res[i] = (np.argmax(result[i]))
            # print res[i]
        # print len(res)
        np.savetxt("results5.csv", res.astype(dtype=int))

alex = AlexNet()
# alex.train()
alex.test()
print utils.compare('../label.csv','results5.csv',2000)