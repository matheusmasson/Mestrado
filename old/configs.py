import os, sys, urllib, gzip

import matplotlib
import lasagne as lsg


matplotlib.use('Agg')

try:
    import cPickle as pickle
except:
    import pickle
sys.setrecursionlimit(10000)

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import MaxPool2DLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum

from lasagne.layers import get_all_params

from lasagne.layers.cuda_convnet import Conv2DCCLayer as Conv2DLayerFast
from lasagne.layers.cuda_convnet import MaxPool2DCCLayer as MaxPool2DLayerFast
from lasagne.layers.cuda_convnet import NINLayer_c01b
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import TrainSplit
from nolearn.lasagne import objective
from nolearn.lasagne.visualize import plot_loss
from nolearn.lasagne.visualize import plot_conv_weights
from nolearn.lasagne.visualize import plot_conv_activity
from nolearn.lasagne.visualize import plot_occlusion

from sklearn.metrics import f1_score

from nolearn.lasagne import PrintLayerInfo

import numpy as np
import theano
import theano.tensor as T
import csv
import h5py
import time
import skimage
import skimage.transform
import sklearn
from scipy import sparse

from sklearn import metrics

from sklearn import preprocessing
from scipy.spatial import distance

from collections import OrderedDict
from datetime import datetime, timedelta
import warnings



conf0 = [
    # layer dealing with the input data
    (InputLayer, {'shape': (None, 1, 128, 128)}),

   (Conv2DLayerFast, {'num_filters': 16, 'filter_size': (3, 3)}),
     (Conv2DLayerFast, {'num_filters': 16, 'filter_size': (3, 3)}),
     (Conv2DLayerFast, {'num_filters': 16, 'filter_size': (3, 3)}),

    (MaxPool2DLayerFast, {'pool_size': (2, 2)}),

    # first stage of our convolutional layers

     (Conv2DLayerFast, {'num_filters': 32, 'filter_size': (3, 3)}),
     (Conv2DLayerFast, {'num_filters': 32, 'filter_size': (3, 3)}),

    (MaxPool2DLayerFast, {'pool_size': (2, 2)}),

    # second stage of our convolutional layers
    (Conv2DLayerFast, {'num_filters': 64, 'filter_size': (3, 3)}),
    (Conv2DLayerFast, {'num_filters': 64, 'filter_size': (3, 3)}),

    #(NINLayer_c01b, {'num_units': 32}),
    (MaxPool2DLayerFast, {'pool_size':  (2, 2)}),

     # third stage of our convolutional layers
    (Conv2DLayerFast, {'num_filters': 128, 'filter_size': (3, 3)}),
        (Conv2DLayerFast, {'num_filters': 128, 'filter_size': (3, 3)}),

    #(NINLayer_c01b, {'num_units': 32}),
    (MaxPool2DLayerFast, {'pool_size':  (2, 2)}),



     # four stage of our convolutional layers
    (Conv2DLayerFast, {'num_filters': 32, 'filter_size': (3, 3)}),

    (MaxPool2DLayerFast, {'pool_size': (2, 2)}),



    # two dense layers with dropout
    (DenseLayer, {'num_units': 64}),
    (DropoutLayer, {}),
    (DenseLayer, {'num_units': 64}),

    # the output layer
    (DenseLayer, {'num_units': 50, 'nonlinearity': softmax}),
]


conf_atual_2d = [
    # layer dealing with the input data
    (InputLayer, {'shape': (None, 1, 128, 128)}),

    (Conv2DLayerFast, {'num_filters': 32, 'filter_size': (3, 3)}),
    (MaxPool2DLayerFast, {'pool_size': (2, 2)}),

    (Conv2DLayerFast, {'num_filters': 32, 'filter_size': (3, 3)}),
    (MaxPool2DLayerFast, {'pool_size': (2, 2)}),

    (Conv2DLayerFast, {'num_filters': 128, 'filter_size': (3, 3)}),
    (MaxPool2DLayerFast, {'pool_size': (2, 2)}),

    (Conv2DLayerFast, {'num_filters': 32, 'filter_size': (3, 3)}),
    (MaxPool2DLayerFast, {'pool_size': (2, 2)}),

    (Conv2DLayerFast, {'num_filters': 16, 'filter_size': (3, 3)}),
    (MaxPool2DLayerFast, {'pool_size': (2, 2)}),

    # two dense layers with dropout
    (DenseLayer, {'num_units': 128}),
    (DropoutLayer, {}),
    (DenseLayer, {'num_units': 128}),

    # the output layer
    (DenseLayer, {'num_units': 50, 'nonlinearity': softmax}),
]


conf_atual_1d = [
    # layer dealing with the input data
    (InputLayer, {'shape': (None, 128, 230)}),

    (lsg.layers.Conv1DLayer, {'num_filters': 32, 'filter_size': 3}),
    (lsg.layers.FeaturePoolLayer, {'pool_size': 2, 'axis': 2}),

    (lsg.layers.Conv1DLayer, {'num_filters': 64, 'filter_size': 3}),
    (lsg.layers.FeaturePoolLayer, {'pool_size': 2, 'axis': 2}),

    (lsg.layers.Conv1DLayer, {'num_filters': 128, 'filter_size': 3}),
     (lsg.layers.Conv1DLayer, {'num_filters': 128, 'filter_size': 3}),
    (lsg.layers.FeaturePoolLayer, {'pool_size': 2, 'axis': 2}),

    (lsg.layers.Conv1DLayer, {'num_filters': 32, 'filter_size': 3}),
    (lsg.layers.FeaturePoolLayer, {'pool_size': 2, 'axis': 2}),

    (lsg.layers.Conv1DLayer, {'num_filters': 16, 'filter_size': 3}),
    (lsg.layers.FeaturePoolLayer, {'pool_size': 2, 'axis': 2}),

    # two dense layers with dropout
    (DenseLayer, {'num_units': 128}),
    (DropoutLayer, {}),
    (DenseLayer, {'num_units': 128}),

    # the output layer
    (DenseLayer, {'num_units': 50, 'nonlinearity': softmax}),
]



net0 = NeuralNet(
    layers=conf_atual_1d,
    max_epochs=20,

    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.975,


    regression=True,

    verbose=3,
)

print "informacoes da rede"
net0.initialize()
layer_info = PrintLayerInfo()
layer_info(net0)
