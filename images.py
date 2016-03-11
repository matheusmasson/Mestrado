import os, sys, urllib, gzip

import matplotlib

matplotlib.use('Agg')
import lasagne as lsg

try:
    import cPickle as pickle
except:
    import pickle
sys.setrecursionlimit(10000)
import gc

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import MaxPool2DLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from lasagne.updates import adam

from lasagne.layers import get_output, Upscale2DLayer, ReshapeLayer

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
import wmf
import networks as nt
import DataBase as db
import metrics
import time
from scipy import sparse
import pandas as pd

DATASET_PATH = "/media/matheus/Files/DataSet/spectrograms_dataset_micro3.h5"
DATASET_FPATH = "/media/matheus/Files/DataSet/factors_small.h5"
NETW_PARAMS = '/media/matheus/Files/aprendizado/teste_img9.pkl'


dataset = h5py.File(DATASET_PATH, 'r')
dfs = h5py.File(DATASET_FPATH, 'r')

song_factors = dfs['factors'][:]

# Get cold index
index_cold = np.where(np.all(song_factors == 0, axis=1))[0]
ds_size = len(song_factors)
list_index = range(0, ds_size)
list_index_not_cold = list(np.delete(np.array(list_index), index_cold))

all_factors = dfs['factors'][:]
all_factors = all_factors[list_index_not_cold]

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
all_factors = min_max_scaler.fit_transform(all_factors)

list_index_not_cold = list_index_not_cold[0:20]

X = dataset["spectrograms"][list_index_not_cold]

y = all_factors[:]
y = np.array(y)

X = X[:, :, 0:256]
num_examples = X.shape[0]
layers = nt.get_network(256, 256, 1)
X = np.array(X).astype(np.float32)
X = X.reshape(num_examples, 1, 256, 256)

net0 = NeuralNet(
    layers=layers,

    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.975,
    regression=True,

    verbose=3,
)

net0.load_params_from(NETW_PARAMS)


print "salvando imagens pesos"

"""
print "peso - layer 1"
file = plot_conv_weights(net0.layers_[1], figsize=(4, 4))
file.savefig('/media/matheus/Files/aprendizado/img/pesos_aprendidos_16f_3x3_l1.png')
print "peso - layer 2"
file = plot_conv_weights(net0.layers_[2], figsize=(4, 4))
file.savefig('/media/matheus/Files/aprendizado/img/pesos_aprendidos_16f_3x3_l2.png')
print "peso - layer 3"
file = plot_conv_weights(net0.layers_[3], figsize=(4, 4))
file.savefig('/media/matheus/Files/aprendizado/img/pesos_aprendidos_16f_3x3_l3.png')
print "peso - layer 5"
file = plot_conv_weights(net0.layers_[5], figsize=(4, 4))
file.savefig('/media/matheus/Files/aprendizado/img/pesos_aprendidos_32f_6x6_l5.png')
print "peso - layer 6"
file = plot_conv_weights(net0.layers_[6], figsize=(4, 4))
file.savefig('/media/matheus/Files/aprendizado/img/pesos_aprendidos_32f_6x6_l6.png')
print "peso - layer 8"
file = plot_conv_weights(net0.layers_[8], figsize=(4, 4))
file.savefig('/media/matheus/Files/aprendizado/img/pesos_aprendidos_64f_12x12_l8.png')
print "peso - layer 9"
file = plot_conv_weights(net0.layers_[9], figsize=(4, 4))
file.savefig('/media/matheus/Files/aprendizado/img/pesos_aprendidos_64f_12x12_l9.png')
print "peso - layer 11"
file = plot_conv_weights(net0.layers_[11], figsize=(4, 4))
file.savefig('/media/matheus/Files/aprendizado/img/pesos_aprendidos_128f_24x24_l11.png')

print "peso - layer 12"
file = plot_conv_weights(net0.layers_[12], figsize=(4, 4))
file.savefig('/media/matheus/Files/aprendizado/img/pesos_aprendidos_128f_24x24_l12.png')

print "peso - layer 14"
file = plot_conv_weights(net0.layers_[14], figsize=(4, 4))
file.savefig('/media/matheus/Files/aprendizado/img/pesos_aprendidos_32f_48x48_l14.png')



print "salvando imagens atividades"

x = X[0:1]
file = plot_conv_activity(net0.layers_[1], x)
file.savefig('/media/matheus/Files/aprendizado/img/atividade_l1_16.png')

file = plot_conv_activity(net0.layers_[2], x)
file.savefig('/media/matheus/Files/aprendizado/img/atividade_l2_16.png')

file = plot_conv_activity(net0.layers_[3], x)
file.savefig('/media/matheus/Files/aprendizado/img/atividade_l3_16.png')

file = plot_conv_activity(net0.layers_[5], x)
file.savefig('/media/matheus/Files/aprendizado/img/atividade_l5_16.png')

file = plot_conv_activity(net0.layers_[6], x)
file.savefig('/media/matheus/Files/aprendizado/img/atividade_l6_16.png')

file = plot_conv_activity(net0.layers_[8], x)
file.savefig('/media/matheus/Files/aprendizado/img/atividade_l8_16.png')

file = plot_conv_activity(net0.layers_[9], x)
file.savefig('/media/matheus/Files/aprendizado/img/atividade_l9_16.png')

file = plot_conv_activity(net0.layers_[11], x)
file.savefig('/media/matheus/Files/aprendizado/img/atividade_l11_16.png')

file = plot_conv_activity(net0.layers_[12], x)
file.savefig('/media/matheus/Files/aprendizado/img/atividade_l12_16.png')

file = plot_conv_activity(net0.layers_[14], x)
file.savefig('/media/matheus/Files/aprendizado/img/atividade_l14_16.png')
"""
print "salvando imagens occlusion"

for i in range(50):
    if i > 30:
        file = plot_occlusion(net0, X[:5], y[:5, i])
        file.savefig('/media/matheus/Files/aprendizado/img/occlusion_' + str(i + 1) +'.png')
        if i % 10 == 0:
            file.close('all')

