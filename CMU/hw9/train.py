'''
Requirements:
sudo pip install pydot
sudo apt-get install -y graphviz
Interesting resources on Caffe:
 - https://github.com/BVLC/caffe/tree/master/examples
 - http://nbviewer.ipython.org/github/joyofdata/joyofdata-articles/blob/master/deeplearning-with-caffe/Neural-Networks-with-Caffe-on-the-GPU.ipynb

Interesting resources on Iris with ANNs:
 - iris data set test bed: http://deeplearning4j.org/iris-flower-dataset-tutorial.html
 - http://se.mathworks.com/help/nnet/examples/iris-clustering.html
 - http://lab.fs.uni-lj.si/lasin/wp/IMIT_files/neural/doc/seminar8.pdf

Synonyms:
 - output = label = target
 - input = feature
'''
import sys
sys.path.append("/home/haodong/caffe/python")

import subprocess
import platform
import copy

import numpy as np
import sklearn
from sklearn.cross_validation import StratifiedShuffleSplit
import matplotlib.pyplot as plt
import h5py
import caffe
import caffe.draw
import load
import scipy.io


def load_data(is_loading_train):
    '''
    Load Iris Data set
    '''
    if is_loading_train:
        file_dir = 'data_mat/data_batch.mat'
        data = scipy.io.loadmat(file_dir)
        labels = data['labels']
        data = data['data']

        new_data = {}
        image1, image2 = load._convert_images(data)

        new_data['input'] = image2
        new_data['output'] = labels
    else:
        file_dir = 'data_mat/test_data.mat'
        data = scipy.io.loadmat(file_dir)
        print data.keys()
        data = data['data']
        new_data = {}
        image1, image2 = load._convert_images(data)
        new_data['input'] = image2
        print image2.shape

    return new_data


def save_data_as_hdf5(hdf5_data_filename, data, isTest = False):
    '''
    HDF5 is one of the data formats Caffe accepts
    '''
    with h5py.File(hdf5_data_filename, 'w') as f:
        f['data'] = data['input'].astype(np.float32)
        if not isTest:
            f['label'] = data['output'].astype(np.float32)


def train(solver_prototxt_filename):
    '''
    Train the ANN
    '''
    caffe.set_mode_gpu()
    solver = caffe.get_solver(solver_prototxt_filename)
    solver.solve()


def print_network_parameters(net):
    '''
    Print the parameters of the network
    '''
    print(net)
    print('net.inputs: {0}'.format(net.inputs))
    print('net.outputs: {0}'.format(net.outputs))
    print('net.blobs: {0}'.format(net.blobs))
    print('net.params: {0}'.format(net.params))


def get_predicted_output(deploy_prototxt_filename, caffemodel_filename, input, net=None):
    '''
    Get the predicted output, i.e. perform a forward pass
    '''
    if net is None:
        net = caffe.Net(deploy_prototxt_filename, caffemodel_filename, caffe.TEST)
    # out = net.forward(data=input)
    out = net.forward()
    # print('out: {0}'.format(out))
    return out[net.outputs[0]]


import google.protobuf


def print_network(prototxt_filename, caffemodel_filename):
    '''
    Draw the ANN architecture
    '''
    _net = caffe.proto.caffe_pb2.NetParameter()
    f = open(prototxt_filename)
    google.protobuf.text_format.Merge(f.read(), _net)
    caffe.draw.draw_net_to_file(_net, prototxt_filename + '.png')
    print('Draw ANN done!')


def print_network_weights(prototxt_filename, caffemodel_filename):
    '''
    For each ANN layer, print weight heatmap and weight histogram
    '''
    net = caffe.Net(prototxt_filename, caffemodel_filename, caffe.TEST)
    for layer_name in net.params:
        # weights heatmap
        arr = net.params[layer_name][0].data
        plt.clf()
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        cax = ax.matshow(arr, interpolation='none')
        fig.colorbar(cax, orientation="horizontal")
        plt.savefig('{0}_weights_{1}.png'.format(caffemodel_filename, layer_name), dpi=100, format='png',
                    bbox_inches='tight')  # use format='svg' or 'pdf' for vectorial pictures
        plt.close()

        # weights histogram
        plt.clf()
        plt.hist(arr.tolist(), bins=20)
        plt.savefig('{0}_weights_hist_{1}.png'.format(caffemodel_filename, layer_name), dpi=100, format='png',
                    bbox_inches='tight')  # use format='svg' or 'pdf' for vectorial pictures
        plt.close()


def get_predicted_outputs(deploy_prototxt_filename, caffemodel_filename, inputs):
    '''
    Get several predicted outputs
    '''
    outputs = []
    net = caffe.Net(deploy_prototxt_filename, caffemodel_filename, caffe.TEST)
    for input in inputs:
        # print(input)
        outputs.append(copy.deepcopy(get_predicted_output(deploy_prototxt_filename, caffemodel_filename, input, net)))
    return outputs


def get_accuracy(true_outputs, predicted_outputs):
    '''

    '''
    number_of_samples = true_outputs.shape[0]
    number_of_outputs = true_outputs.shape[1]
    threshold = 0.0  # 0 if SigmoidCrossEntropyLoss ; 0.5 if EuclideanLoss
    for output_number in range(number_of_outputs):
        predicted_output_binary = []
        for sample_number in range(number_of_samples):
            # print(predicted_outputs)
            # print(predicted_outputs[sample_number][output_number])
            if predicted_outputs[sample_number][0][output_number] < threshold:
                predicted_output = 0
            else:
                predicted_output = 1
            predicted_output_binary.append(predicted_output)

        print(
        'accuracy: {0}'.format(sklearn.metrics.accuracy_score(true_outputs[:, output_number], predicted_output_binary)))
        print(sklearn.metrics.confusion_matrix(true_outputs[:, output_number], predicted_output_binary))


def main():
    '''
    This is the main function
    '''

    # Set parameters
    solver_prototxt_filename = 'cifar_solver.prototxt'
    train_test_prototxt_filename = 'cifar_train_test.prototxt'
    # train_test_prototxt_filename = '1-nodata.prototxt'
    # deploy_prototxt_filename = 'cifar_deploy.prototxt'
    # deploy_prototxt_batch2_filename = 'cifar_deploy_batchsize2.prototxt'
    hdf5_train_data_filename = 'cifar_train_data.hdf5'
    hdf5_test_data_filename = 'cifar_test_data.hdf5'
    caffemodel_filename = 'cifar3_iter_6000.caffemodel.h5'  # generated by train()

    is_trained = False
    is_testing = True
    # Prepare data
    if not is_trained:
        train_data = load_data(True)
        # print(data)
        save_data_as_hdf5(hdf5_train_data_filename, train_data)
        # Train network
        train(solver_prototxt_filename)
    if is_testing:
        test_data = load_data(False)
        save_data_as_hdf5(hdf5_test_data_filename, test_data, True)

    # Get predicted outputs
    # input = np.array([[5.1, 3.5, 1.4, 0.2]])
    result = get_predicted_output(train_test_prototxt_filename, caffemodel_filename, input)
    print len(result)
    res = np.zeros(len(result),dtype=int)
    for i in xrange(len(result)):
         res[i] = (int(np.argmax(result[i])))
         print res[i]
    print 'length of result'
    print len(res)
    np.savetxt("output.csv", res)
    # input = np.array([[[[5.1, 3.5, 1.4, 0.2]]], [[[5.9, 3., 5.1, 1.8]]]])
    print 'real'
    # print test_data['output'][0:99]
    # Print network
    #print_network(train_test_prototxt_filename, caffemodel_filename)
    #print_network(train_test_prototxt_filename, caffemodel_filename)
    #print_network_weights(train_test_prototxt_filename, caffemodel_filename)

    # Compute performance metrics
    # inputs = input = np.array([[[[ 5.1,  3.5,  1.4,  0.2]]],[[[ 5.9,  3. ,  5.1,  1.8]]]])
    # inputs = data['input']
    # outputs = get_predicted_outputs(deploy_prototxt_filename, caffemodel_filename, inputs)
    # get_accuracy(data['output'], outputs)


if __name__ == "__main__":
    main()
    # cProfile.run('main()') # if you want to do some profiling