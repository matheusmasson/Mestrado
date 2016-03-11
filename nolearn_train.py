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

import networks as nt

warnings.filterwarnings('ignore', module='lasagne')

DATASET_PATH = "/media/matheus/Files/DataSet/spectrograms_dataset_micro3.h5"
DATASET_FPATH = "/media/matheus/Files/DataSet/factors_small.h5"

net_runs = [1,4,5]

NUM_FACTORS = 50
CHUNK_SIZE = 2048
NUM_CHUNKS = 2
LEARNING_RATE = 0.01 # 0.01
MOMENTUM = 0.9
NUM_EXAMPLES_EVAL_USED = 758


d = h5py.File(DATASET_PATH, 'r')
dfs = h5py.File(DATASET_FPATH, 'r')

max = 9000
min_train = 0
min_eval = 8231
max_eval = 8990


song_factors = dfs['factors'][:]

# Get cold index
index_cold = np.where(np.all(song_factors == 0, axis=1))[0]
ds_size = len(song_factors)
list_index = range(0, ds_size)
list_index_not_cold = list(np.delete(np.array(list_index), index_cold))


print "Comecou a ler o arquivo"

print "vai ler os specs"
data_train = d["spectrograms"][:]
data_train = data_train[list_index_not_cold]
#data_train = data_train[:1280]
print "leu todos"
data_eval = data_train[min_eval:max_eval]
#data_eval = data_train[900:1028]
print "copiou os de validacao"

print "vai ler os fatores"
# Get the factors and scale all the values between 0 and 1 for convnet performance
all_factors = dfs['factors'][:]
all_factors = all_factors[list_index_not_cold]



print "leu todos os fatores"



WMF_FEATURES = '/media/matheus/Files/DataSet/testesmall.npz'
wmf_file = np.load(WMF_FEATURES)
training_user_facts = wmf_file['U']

#min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
#all_factors = min_max_scaler.fit_transform(all_factors)

"""
C = np.concatenate((all_factors, training_user_facts), axis=0)


min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
C = min_max_scaler.fit_transform(C)

qtds = len(all_factors)
all_factors = C[0:qtds]
"""

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
all_factors = min_max_scaler.fit_transform(all_factors)

#rs = preprocessing.MaxAbsScaler()
#all_factors = rs.fit_transform(all_factors)

factors_norm = preprocessing.normalize(all_factors, norm='l2')



labels_train = all_factors[:]
labels_eval = all_factors[min_eval:max_eval]
#labelsnorm = factors_norm[:]
labelsnorm = all_factors[:]


CHANGE_RATE = True


def on_epoch_finish_callback2(net, hist):
    epc_atual = len(hist)

    if epc_atual == 4:
        net.update_learning_rate = 0.001

    if epc_atual == 6:
        net.update_learning_rate = 0.0001


def on_epoch_finish_callback(net, hist):
    epc_atual = len(hist)

    if epc_atual == 12:
        net.update_learning_rate = 0.001

    if epc_atual == 14:
        net.update_learning_rate = 0.0001

    #if epc_atual % 10 == 0:
    #    print "Rate: ", net.update_learning_rate,  " Memoria livre na gpu: ", theano.sandbox.cuda.cuda_ndarray.cuda_ndarray.mem_info()[0] / 1024.**2



if 1 in net_runs:
    print "teste rede img2"
    print "1.8..- nova rede cortada sem transformar, 256 de tempo"
    X = data_train[:, :, 0:256]
    y = all_factors
    num_examples = X.shape[0]

    layers = nt.get_network(256, 256, 1)
    X = np.array(X).astype(np.float32)
    y = np.array(y).astype(np.float32)
    X = X.reshape(num_examples, 1, 256, 256)

    net1 = NeuralNet(
        layers=layers,
        max_epochs=NUM_CHUNKS,

        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.9,

        on_epoch_finished = [on_epoch_finish_callback2],

        regression=True,

        verbose=3,
    )

    net1.batch_iterator_train.batch_size = 64
    net1.train_split.eval_size = 0
    #net1.batch_iterator_train.shuffle = True
    net1.fit(X, y)

    y = np.array(labelsnorm).astype(np.float32)
    net1.max_epochs = 5

    print "Dados Normalizados:"
    net1.fit(X, y)

    #net1.save_params_to('/media/matheus/Files/aprendizado/conv_test_net_cnn1_w.pkl')
    #file = plot_loss(net1)
    #file.savefig('/media/matheus/Files/aprendizado/grafico_taxa_perda_cnn1.png')

    net1.save_params_to('/media/matheus/Files/aprendizado/teste_img10.pkl')
    file = plot_loss(net1)
    file.savefig('/media/matheus/Files/aprendizado/teste_img10.png')

if 2 in net_runs:
    print "2- rede atual 2 1d "
    X = data_train[:, :, 0:256]
    y = labels_train
    num_examples = X.shape[0]

    layers = nt.get_network(256, 256, 2)
    X = np.array(X).astype(np.float32)
    y = np.array(y).astype(np.float32)
    X = X.reshape(num_examples, 256, 256)

    epcs = NUM_CHUNKS


    net2 = NeuralNet(
        layers=layers,
        max_epochs=NUM_CHUNKS,
        on_epoch_finished = [on_epoch_finish_callback],

        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.9,


        regression=True,

        verbose=3,
    )
    #net2.batch_iterator_train.batch_size = CHUNK_SIZE
    #net2.batch_iterator_train.shuffle = True
    net2.fit(X, y)

    y = np.array(labelsnorm).astype(np.float32)
    net2.max_epochs = 3

    print "Dados Normalizados:"
    net2.fit(X, y)

    net2.save_params_to('/media/matheus/Files/aprendizado/2teste_img10.pkl')
    file = plot_loss(net2)
    file.savefig('/media/matheus/Files/aprendizado/2teste_img10.png')


if 3 in net_runs:
    print "3- rede teste 3"
    X = data_train[:, :, 0:256]
    y = labels_train
    num_examples = X.shape[0]

    layers = nt.get_network(256, 256, 3)
    X = np.array(X).astype(np.float32)
    y = np.array(y).astype(np.float32)
    X = X.reshape(num_examples, 1, 256, 256)
    net3 = NeuralNet(
        layers=layers,
        max_epochs=NUM_CHUNKS,

        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.9,

        on_epoch_finished = [on_epoch_finish_callback],


        regression=True,

        verbose=3,
    )
    net3.batch_iterator_train.batch_size = 64

    #net3.batch_iterator_train.shuffle = True
    net3.fit(X, y)

    y = np.array(labelsnorm).astype(np.float32)
    net3.max_epochs = 3

    print "Dados Normalizados:"
    net3.fit(X, y)

    net3.save_params_to('/media/matheus/Files/aprendizado/3teste_img10.pkl')
    file = plot_loss(net3)
    file.savefig('/media/matheus/Files/aprendizado/3teste_img10.png')


if 4 in net_runs:
    print "4- rede teste 4"
    X = data_train[:, :, 0:256]
    y = labels_train
    num_examples = X.shape[0]

    layers = nt.get_network(256, 256, 4)
    X = np.array(X).astype(np.float32)
    y = np.array(y).astype(np.float32)
    X = X.reshape(num_examples, 1, 256, 256)
    net4 = NeuralNet(
        layers=layers,
        max_epochs=NUM_CHUNKS,

        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.9,

        on_epoch_finished = [on_epoch_finish_callback],


        regression=True,

        verbose=3,
    )
    net4.batch_iterator_train.batch_size = 64
    #net3.batch_iterator_train.shuffle = True
    net4.train_split.eval_size = 0
    net4.fit(X, y)

    y = np.array(labelsnorm).astype(np.float32)
    net4.max_epochs = 13

    print "Dados Normalizados:"
    net4.fit(X, y)

    net4.save_params_to('/media/matheus/Files/aprendizado/4teste_img10.pkl')
    file = plot_loss(net4)
    file.savefig('/media/matheus/Files/aprendizado/4teste_img10.png')

if 5 in net_runs:
    print "5- rede teste 6"
    X = data_train[:, :, 0:256]
    y = labels_train
    num_examples = X.shape[0]

    layers = nt.get_network(256, 256, 6)
    X = np.array(X).astype(np.float32)
    y = np.array(y).astype(np.float32)
    X = X.reshape(num_examples, 1, 256, 256)
    net6 = NeuralNet(
        layers=layers,
        max_epochs=NUM_CHUNKS,

        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.9,

        on_epoch_finished = [on_epoch_finish_callback],


        regression=True,

        verbose=3,
    )
    net6.batch_iterator_train.batch_size = 64
    #net3.batch_iterator_train.shuffle = True
    net6.train_split.eval_size = 0
    net6.fit(X, y)

    y = np.array(labelsnorm).astype(np.float32)
    net6.max_epochs = 13

    print "Dados Normalizados:"
    net6.fit(X, y)

    net6.save_params_to('/media/matheus/Files/aprendizado/5teste_img10.pkl')
    file = plot_loss(net6)
    file.savefig('/media/matheus/Files/aprendizado/5teste_img10.png')








