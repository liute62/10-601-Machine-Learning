import pickle
import cnn_lenet
import testLeNet

fileObject = open('lenet.mat','r')
params = pickle.load(fileObject)
layers = testLeNet.get_lenet()

fullset = False
xtrain, ytrain, xval, yval, xtest, ytest = cnn_lenet.load_mnist(fullset)

layers[1]['batch_size'] = xtest.shape[1]
cptest, _ = cnn_lenet.conv_net(params, layers, xtest, ytest)
