import os, sys, urllib, gzip

import matplotlib

matplotlib.use('Agg')
matplotlib.rcParams['figure.max_open_warning'] = 1000
import lasagne as lsg

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
from lasagne.updates import adam

from lasagne.layers import get_output, Upscale2DLayer, ReshapeLayer

from lasagne.layers import get_all_params

from lasagne.layers.cuda_convnet import Conv2DCCLayer as Conv2DLayerFast
from lasagne.layers.cuda_convnet import MaxPool2DCCLayer as MaxPool2DLayerFast
from lasagne.layers.cuda_convnet import NINLayer_c01b
from nolearn.lasagne import NeuralNet, BatchIterator, TrainSplit
from nolearn.lasagne import TrainSplit
from nolearn.lasagne import objective
from nolearn.lasagne.visualize import plot_loss
from nolearn.lasagne.visualize import plot_conv_weights
from nolearn.lasagne.visualize import plot_conv_activity
from nolearn.lasagne.visualize import plot_occlusion
from lasagne.objectives import aggregate
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
import gc
from sklearn import metrics

from sklearn import preprocessing
from scipy.spatial import distance

from collections import OrderedDict
from datetime import datetime, timedelta
import warnings
import pandas as pd

import networks as nt
import DataBase as db
import wmf
import metrics as mtfull

warnings.filterwarnings('ignore', module='lasagne')

DATASET_PATH = "/media/matheus/Files/DataSet/spectrograms_dataset_full.h5"
DATASET_FPATH = "/media/matheus/Files/DataSet/factors_full.h5"
WMF_FEATURES = '/media/matheus/Files/DataSet/testefull.npz'

d = h5py.File(DATASET_PATH, 'r')
dfs = h5py.File(DATASET_FPATH, 'r')

print "18+++======="
song_factors = dfs['factors'][:]

# Get cold index
index_cold = np.where(np.all(song_factors == 0, axis=1))[0]
ds_size = len(song_factors)
list_index = range(0, ds_size)
list_index_not_cold = list(np.delete(np.array(list_index), index_cold))


def on_epoch_finish_callback_m1(net, hist):
    epc_atual = len(hist)

    if epc_atual > 10:
        net.update_learning_rate = 0.001

    if epc_atual > 12:
        net.update_learning_rate = 0.0001


print "1* Modelo1- Treino da CNN no dataset completo - funcao objetivo 1 (predicao dos fatores)"

print "vai ler os fatores"
# Get the factors and scale all the values between 0 and 1 for convnet performance
all_factors = dfs['factors'][:]
all_factors = all_factors[list_index_not_cold]
all_ids = dfs['id'][list_index_not_cold]

print "leu todos os fatores"
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
all_factors = min_max_scaler.fit_transform(all_factors)
labels_train = all_factors[:]
labelsnorm = preprocessing.normalize(all_factors, norm='l2')

layers = nt.get_network(256, 256, 4)

net1 = NeuralNet(
    layers=layers,
    max_epochs=1,

    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,
    train_split=TrainSplit(eval_size=0),
    batch_iterator_train=BatchIterator(batch_size=64, shuffle=True),

    on_epoch_finished = [on_epoch_finish_callback_m1],

    regression=True,

    verbose=3,
)

y = labels_train
y = np.array(y).astype(np.float32)

num_examples = y.shape[0]
print "tamanho do treino: ", num_examples
batchsize = num_examples // 4


def get_batch(net, bnum, norm):
    ini = bnum * batchsize
    end = ini + batchsize

    if not norm:
        y_batch = labels_train[ini:end]
    else:
        y_batch = labelsnorm[ini:end]

    y_batch = np.array(y_batch).astype(np.float32)

    rg_not_cold = list_index_not_cold[ini:end]

    x_batch = d["spectrograms"][rg_not_cold]
    x_batch = x_batch[:, :, 0:256]
    size = x_batch.shape[0]
    x_batch = x_batch.reshape(size, 1, 256, 256)

    net.fit(x_batch, y_batch)
    del x_batch, y_batch
    gc.collect()

for i in range((num_examples + batchsize - 1) // batchsize):
    get_batch(net1, i, False)
    gc.collect()

for j in xrange(10):
    for i in xrange((num_examples + batchsize - 1) // batchsize):
        get_batch(net1, i, True)
        gc.collect()


net1.save_params_to('/media/matheus/Files/aprendizado/conv_full1.pkl')
file = plot_loss(net1)
file.savefig('/media/matheus/Files/aprendizado/conv_full1.png')
