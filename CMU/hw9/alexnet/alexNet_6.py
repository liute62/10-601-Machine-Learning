import sys
sys.path.append("/home/haodong/caffe/examples/cifar3/workspace")
import utils
import const
sys.path.append("/home/haodong/caffe/python")
import caffe
import copy
import numpy as np
import matplotlib.pyplot as plt
import math

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
        # raw = train_data['input']
        # r = raw[0]
        # g = raw[1]
        # b = raw[2]
        # print len(r.flatten())
        # r_std = np.std(r.flatten())
        # r_mean = np.mean(r.flatten())
        # adjusted_stddev = max(r_std, 1.0 / math.sqrt(3072))
        # r = (np.array(r, dtype=np.float32) - r_mean) / adjusted_stddev
        #
        # g_std = np.std(g.flatten())
        # g_mean = np.mean(g.flatten())
        # adjusted_stddev = max(g_std, 1.0 / math.sqrt(3072))
        # g = (np.array(g, dtype=np.float32) - g_mean) / adjusted_stddev
        #
        #
        # b_std = np.std(b.flatten())
        # b_mean = np.mean(b.flatten())
        # adjusted_stddev = max(b_std, 1.0 / math.sqrt(3072))
        # b = (np.array(b, dtype=np.float32) - b_mean) / adjusted_stddev
        #
        # raw[0] = r
        # raw[1] = g
        # raw[2] = b
        # print raw
        # train_data['input'] = raw
        # # train_data['input'] = utils.additive_gaussian_noise(raw)
        # print train_data['input']
        # utils.save_data_as_lmdb('cifar6_train_data_lmdb',train_data)
        caffe.set_mode_gpu()
        solver = caffe.get_solver('alexnet_solver_6.prototxt')
        solver.solve()
        pass

    def test(self):
        test_data, __ = utils.load_test_data()
        # raw = test_data['input']
        # test_data['input'] = np.array(raw, dtype=np.float32) / 255.0
        # utils.save_data_as_lmdb('cifar6_test_data_lmdb', test_data, True)
        result = self.__get_predicted_output('alexnet_result_6.prototxt', 'cifar3_6_iter_120000.caffemodel.h5')
        res = np.zeros(len(result), dtype=int)
        for i in xrange(len(result)):
            res[i] = (np.argmax(result[i]))
            # print res[i]
        # print len(res)
        np.savetxt("results6.csv", res.astype(dtype=int))

alex = AlexNet()
# alex.train()
alex.test()
print utils.compare('../label.csv','results6.csv',2000)