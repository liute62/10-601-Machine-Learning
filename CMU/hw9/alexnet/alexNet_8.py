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
        # print train_data['input'].shape
        # print train_data['output'].shape
        # train_data['input'] = (np.array(train_data['input'], dtype=np.float32))
        # train_data['input'] = utils.additive_gaussian_noise(train_data['input'])
        # print train_data['input']
        # utils.save_data_as_lmdb('cifar7_train_data_lmdb',train_data)
        caffe.set_mode_gpu()
        solver = caffe.get_solver('alexnet_solver_8.prototxt')
        solver.solve()
        pass

    def test(self,str):
        test_data, __ = utils.load_test_data()
        # utils.save_data_as_lmdb('cifar5_test_data_lmdb', test_data, True)
        result = self.__get_predicted_output('alexnet_result_8.prototxt', 'cifar3_8_iter_'+str+'.caffemodel.h5')
        res = np.zeros(len(result), dtype=int)
        for i in xrange(len(result)):
            res[i] = (np.argmax(result[i]))
            # print res[i]
        # print len(res)
        np.savetxt("results8.csv", res.astype(dtype=int))

# max_prex = 0
# max_num = 0
# prefix = 80000
# #71000 0.9475
# for i in range(0,20):
#     prefix += 1000
#     alex = AlexNet()
#     alex.test(str(prefix))
#     res = utils.compare('../label.csv','results7.csv',2000)
#     if res > max_num:
#         max_num = res
#         max_prex = prefix
#
# print max_prex,max_num

alex = AlexNet()
# alex.train()
alex.test('180000')
print utils.compare('../label.csv', 'results8.csv', 2000)