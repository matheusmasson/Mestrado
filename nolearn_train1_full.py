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

DATASET_PATH = "/media/matheus/Files/DataSet/spectrograms_dataset_full.h5"
DATASET_FPATH = "/media/matheus/Files/DataSet/factors_full.h5"

net_runs = [1]

NUM_FACTORS = 50
NUM_CHUNKS = 2

d = h5py.File(DATASET_PATH, 'r')
dfs = h5py.File(DATASET_FPATH, 'r')


song_factors = dfs['factors'][:]

# Get cold index
index_cold = np.where(np.all(song_factors == 0, axis=1))[0]
ds_size = len(song_factors)
list_index = range(0, ds_size)
list_index_not_cold = list(np.delete(np.array(list_index), index_cold))


print "Comecou a ler o arquivo"

print "vai ler os specs"
X = d["spectrograms"][list_index_not_cold]
print "leu todos"
print "copiou os de validacao"

print "vai ler os fatores"
# Get the factors and scale all the values between 0 and 1 for convnet performance
all_factors = dfs['factors'][:]
all_factors = all_factors[list_index_not_cold]
print "leu todos os fatores"


min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
all_factors = min_max_scaler.fit_transform(all_factors)

labels_train = all_factors[:]


def on_epoch_finish_callback(net, hist):
    epc_atual = len(hist)

    if epc_atual > 5:
        net.update_learning_rate = 0.001

    if epc_atual % 10 == 0:
        print "Rate: ", net.update_learning_rate,  " Memoria livre na gpu: ", theano.sandbox.cuda.cuda_ndarray.cuda_ndarray.mem_info()[0] / 1024.**2


if 1 in net_runs:
    print "1- Treino da CNN no dataset completo 18"
    X = X[:, :, 0:128]
    y = labels_train

    num_examples = X.shape[0]
    print "tamanho do treino: ", num_examples

    layers = nt.get_network(128, 128, 1)
    X = np.array(X).astype(np.float32)
    y = np.array(y).astype(np.float32)
    X = X.reshape(num_examples, 1, 128, 128)

    net0 = NeuralNet(
        layers=layers,
        max_epochs=NUM_CHUNKS,

        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.975,
        train_split=TrainSplit(eval_size=0),
        batch_iterator_train=BatchIterator(batch_size=128, shuffle=True),

        on_epoch_finished = [on_epoch_finish_callback],

        regression=True,

        verbose=3,
    )
    #net1.batch_iterator_train.batch_size = CHUNK_SIZE
    #net1.batch_iterator_train.shuffle = True
    net0.fit(X, y)

    #net1.save_params_to('/media/matheus/Files/aprendizado/conv_test_net_cnn1_full_w.pkl')
    #file = plot_loss(net1)
    #file.savefig('/media/matheus/Files/aprendizado/grafico_taxa_perda_cnn1_full.png')

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
    """
    print "peso - layer 12"
    file = plot_conv_weights(net0.layers_[12], figsize=(4, 4))
    file.savefig('/media/matheus/Files/aprendizado/img/pesos_aprendidos_128f_24x24_l12.png')

    print "peso - layer 14"
    file = plot_conv_weights(net0.layers_[14], figsize=(4, 4))
    file.savefig('/media/matheus/Files/aprendizado/img/pesos_aprendidos_32f_48x48_l14.png')



    print "salvando imagens atividades"

    for i in range(10):
        x = X[i:i+1]
        file = plot_conv_activity(net0.layers_[1], x)
        file.savefig('/media/matheus/Files/aprendizado/img/temp1' + str(i) + '.png')

        file = plot_conv_activity(net0.layers_[2], x)
        file.savefig('/media/matheus/Files/aprendizado/img/temp2' + str(i) + '.png')

        file = plot_conv_activity(net0.layers_[3], x)
        file.savefig('/media/matheus/Files/aprendizado/img/temp3' + str(i) + '.png')

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

    print "salvando imagens occlusion"

    for i in range(50):
        file = plot_occlusion(net0, X[:5], y[:5, i])
        file.savefig('/media/matheus/Files/aprendizado/img/occlusion_' + str(i + 1) +'.png')
        if i % 10 == 0:
            file.close('all')

if 2 in net_runs:
    print "2- rede atual 1d cortada nao transformando, 214 de tempo"
    X = data_train[:, :, 0:214]
    y = labels_train

    layers = nt.get_network(128, 214, 2)
    X = np.array(X).astype(np.float32)
    y = np.array(y).astype(np.float32)
    X = X.reshape(11044, 128, 214)

    epcs = NUM_CHUNKS


    net2 = NeuralNet(
        layers=layers,
        max_epochs=NUM_CHUNKS,
        on_epoch_finished = [on_epoch_finish_callback],

        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.975,


        regression=True,

        verbose=3,
    )
    #net2.batch_iterator_train.batch_size = CHUNK_SIZE
    #net2.batch_iterator_train.shuffle = True
    net2.fit(X, y)

    net2.save_params_to('/media/matheus/Files/aprendizado/conv_test_net_cnn2_w.pkl')
    file = plot_loss(net2)
    file.savefig('/media/matheus/Files/aprendizado/grafico_taxa_perda_cnn2.png')


if 3 in net_runs:
    print "3- rede teste 2, 128 de tempo cortada e nao transformada"
    X = data_train[:, :, 0:128]
    y = labels_train

    layers = nt.get_network(128, 128, 3)
    X = np.array(X).astype(np.float32)
    y = np.array(y).astype(np.float32)
    X = X.reshape(11044, 1, 128, 128)
    net3 = NeuralNet(
        layers=layers,
        max_epochs=NUM_CHUNKS,

        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.975,

        on_epoch_finished = [on_epoch_finish_callback],


        regression=True,

        verbose=3,
    )
    #net3.batch_iterator_train.batch_size = CHUNK_SIZE
    #net3.batch_iterator_train.shuffle = True
    net3.fit(X, y)

    net3.save_params_to('/media/matheus/Files/aprendizado/conv_test_net_cnn3_w.pkl')
    file = plot_loss(net3)
    file.savefig('/media/matheus/Files/aprendizado/grafico_taxa_perda_cnn3.png')








