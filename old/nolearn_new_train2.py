import os, sys, urllib, gzip

import matplotlib

matplotlib.use('Agg')
matplotlib.rcParams['figure.max_open_warning'] = 1000


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

warnings.filterwarnings('ignore', module='lasagne')

TARGET_PATH = "/media/matheus/Files/DataSet/validacao.csv"
DATASET_PATH = "/media/matheus/Files/DataSet/spectrograms_dataset_micro.h5"
DATASET_FPATH = "/media/matheus/Files/DataSet/factors_small.h5"

SAVE_IMGS = True

NUM_FACTORS = 50
CHUNK_SIZE = 5120
NUM_CHUNKS = 2
#NUM_TIMESTEPS_AUG = 128 # 110
MB_SIZE = 128
LEARNING_RATE = 0.01 # 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0
EVALUATE_EVERY = 1 # always validate since it's fast enough
# SOFTMAX_LAMBDA = 0.01
COMPRESSION_CONSTANT = 10000
#NUM_FREQ_COMPONENTS_AUG = 128
NUM_EXAMPLES_EVAL_USED = 758
TEST_CHUNK = 512

def hms(seconds):
    seconds = np.floor(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    return "%02d:%02d:%02d" % (hours, minutes, seconds)

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

# Remove from dataset cold samples
#song_factors = song_factors[list_index_not_cold]


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


min_max_scaler = preprocessing.MinMaxScaler()
all_factors = min_max_scaler.fit_transform(all_factors)

labels_train = all_factors[:]
labels_eval = all_factors[min_eval:max_eval]


num_examples_train, num_freq_components, num_timesteps = data_train.shape
num_batches_train = CHUNK_SIZE // MB_SIZE

num_examples_eval = data_eval.shape[0]







def fast_warp(img, tf, output_shape, mode='reflect'):
    return skimage.transform._warps_cy._warp_fast(img, tf._matrix, output_shape=output_shape, mode=mode)

def make_chunks(data, labels, freq_aug, tmg_aug):

    tfs_fixed = []
    for offset_time in np.linspace(0, num_timesteps - tmg_aug, 8):
        for offset_freq in np.linspace(0, num_freq_components - freq_aug, 3):
            tfs_fixed.append(skimage.transform.AffineTransform(translation=(offset_time, offset_freq)))

    num_tfs_fixed = len(tfs_fixed) - 1

    chunk = np.empty((len(data), freq_aug, tmg_aug), dtype='float32')
    chunk_labels = np.empty((len(data_train), NUM_FACTORS), dtype='float32')

    for k in xrange(len(data_train) ):
        for l, tf in enumerate(tfs_fixed):
            if (k * num_tfs_fixed + l) < len(chunk) - 1:
                out = fast_warp(data_train[k], tf, output_shape=(freq_aug, tmg_aug), mode='reflect').astype('float32')
                chunk[k * num_tfs_fixed + l] = out
                chunk_labels[k * num_tfs_fixed + l] = labels[k]

    chunk = np.log(1 + COMPRESSION_CONSTANT*chunk) # compression

    return chunk, chunk_labels


def make_data_input(data, labels, num_freq, num_ts, cut, transform):
    print data[0]
    if cut:
        data = data[:, :, num_ts]

    print data[0]

    if transform:
        return make_chunks(data, labels, num_freq, num_ts)

    return data, labels


# Teste com
X, y = make_data_input(data_train, labels_train, 128, 128, False, True)

X = np.array(X).astype(np.float32)
y = np.array(y).astype(np.float32)
X = X.reshape(11044, 1, 128, 128)





## architecture
# 50 <=(3)= 52 <=[2]= 104 <=(3)= 106 <=[2]= 212 <=(3)= 214 <=[2]= 428 <=(3)= 430
# 10 <=(3)= 12 <=[2]= 24 <=(3)= 26 <=[2]= 52 <=(3)= 54 <=[2]= 108 <=(3)= 110
# 25 <=(3)= 27 <=[2]= 54 <=(3)= 56 <=[2]= 112 <=(3)= 114 <=[2]= 228 <=(3)= 230
# 25 <=(3)= 27 <=[2]= 54 <=(3)= 56 <=[2]= 112 <=(3)= 114 <=[2]= 228 <=(3)= 230
# 50 <=(3)= 52 <=[2]= 104 <=(3)= 106 <=[2]= 212 <=(3)= 214

# 26 <=(3)= 52 <=(3)= 54 <=[2]= 108 <=(3)= 110 <=[2]= 220 <=(3)= 222


layers0 = [
    # layer dealing with the input data
    (InputLayer, {'shape': (None, X.shape[1], X.shape[2], X.shape[3])}),

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
    (DenseLayer, {'num_units': NUM_FACTORS, 'nonlinearity': softmax}),
]


def regularization_objective(layers, lambda1=0., lambda2=0., *args, **kwargs):
    # default loss
    losses = objective(layers, *args, **kwargs)
    # get the layers' weights, but only those that should be regularized
    # (i.e. not the biases)
    weights = get_all_params(layers[-1], regularizable=True)
    # sum of absolute weights for L1
    sum_abs_weights = sum([abs(w).sum() for w in weights])
    # sum of squared weights for L2
    sum_squared_weights = sum([(w ** 2).sum() for w in weights])
    # add weights to regular loss
    losses += lambda1 * sum_abs_weights + lambda2 * sum_squared_weights
    return losses

net0 = NeuralNet(
    layers=layers0,
    max_epochs=NUM_CHUNKS,

    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.975,


    regression=True,

    verbose=3,
)

print "Treino"
net0.fit(X, y)
print "terminou treino"

pickle.dump(net0, open('/media/matheus/Files/aprendizado/conv_test_net.pkl','wb'))
net0.save_params_to('/media/matheus/Files/aprendizado/conv_test_w.pkl')

if SAVE_IMGS:

    print "salvou os parametros"
    print "salvando imagem perda"

    file = plot_loss(net0)
    file.savefig('/media/matheus/Files/aprendizado/grafico_taxa_perda.png')


    print "salvando imagens pesos"

    file = plot_conv_weights(net0.layers_[1], figsize=(4, 4))
    file.savefig('/media/matheus/Files/aprendizado/pesos_aprendidos_16f_3x3_l1.png')

    file = plot_conv_weights(net0.layers_[2], figsize=(4, 4))
    file.savefig('/media/matheus/Files/aprendizado/pesos_aprendidos_16f_3x3_l2.png')

    file = plot_conv_weights(net0.layers_[3], figsize=(4, 4))
    file.savefig('/media/matheus/Files/aprendizado/pesos_aprendidos_16f_3x3_l3.png')

    file = plot_conv_weights(net0.layers_[5], figsize=(4, 4))
    file.savefig('/media/matheus/Files/aprendizado/pesos_aprendidos_32f_6x6_l5.png')

    file = plot_conv_weights(net0.layers_[6], figsize=(4, 4))
    file.savefig('/media/matheus/Files/aprendizado/pesos_aprendidos_32f_6x6_l6.png')

    file = plot_conv_weights(net0.layers_[8], figsize=(4, 4))
    file.savefig('/media/matheus/Files/aprendizado/pesos_aprendidos_64f_12x12_l8.png')

    file = plot_conv_weights(net0.layers_[9], figsize=(4, 4))
    file.savefig('/media/matheus/Files/aprendizado/pesos_aprendidos_64f_12x12_l9.png')

    file = plot_conv_weights(net0.layers_[11], figsize=(4, 4))
    file.savefig('/media/matheus/Files/aprendizado/pesos_aprendidos_128f_24x24_l11.png')

    file = plot_conv_weights(net0.layers_[12], figsize=(4, 4))
    file.savefig('/media/matheus/Files/aprendizado/pesos_aprendidos_128f_24x24_l12.png')

    file = plot_conv_weights(net0.layers_[14], figsize=(4, 4))
    file.savefig('/media/matheus/Files/aprendizado/pesos_aprendidos_32f_48x48_l14.png')



    print "salvando imagens atividades"

    x = X[0:1]
    file = plot_conv_activity(net0.layers_[1], x)
    file.savefig('/media/matheus/Files/aprendizado/atividade_l1_16.png')

    file = plot_conv_activity(net0.layers_[2], x)
    file.savefig('/media/matheus/Files/aprendizado/atividade_l2_16.png')

    file = plot_conv_activity(net0.layers_[3], x)
    file.savefig('/media/matheus/Files/aprendizado/atividade_l3_16.png')

    file = plot_conv_activity(net0.layers_[5], x)
    file.savefig('/media/matheus/Files/aprendizado/atividade_l5_16.png')

    file = plot_conv_activity(net0.layers_[6], x)
    file.savefig('/media/matheus/Files/aprendizado/atividade_l6_16.png')

    file = plot_conv_activity(net0.layers_[8], x)
    file.savefig('/media/matheus/Files/aprendizado/atividade_l8_16.png')

    file = plot_conv_activity(net0.layers_[9], x)
    file.savefig('/media/matheus/Files/aprendizado/atividade_l9_16.png')

    file = plot_conv_activity(net0.layers_[11], x)
    file.savefig('/media/matheus/Files/aprendizado/atividade_l11_16.png')

    file = plot_conv_activity(net0.layers_[12], x)
    file.savefig('/media/matheus/Files/aprendizado/atividade_l12_16.png')

    file = plot_conv_activity(net0.layers_[14], x)
    file.savefig('/media/matheus/Files/aprendizado/atividade_l14_16.png')

    print "salvando imagens occlusion"

    for i in range(50):
        file = plot_occlusion(net0, X[:10], y[:10][i])
        file.savefig('/media/matheus/Files/aprendizado/occlusion_' + str(i + 1) +'.png')
        if i % 10 == 0:
            file.close('all')


print "informacoes da rede"
net0.initialize()
layer_info = PrintLayerInfo()
layer_info(net0)

"""
print "validacao"

net0 = pickle.load(open('mnist/conv_ae.pkl','rb'))


X_val, y_val = make_chunks(data_eval, labels_eval)


X_val = np.array(X_val).astype(np.float32)
y_val = np.array(y_val).astype(np.float32)

X_val = X_val.reshape(NUM_EXAMPLES_EVAL_USED, 1, 128, 128)

y_pred = net0.predict(X_val)

print "tamanho validacao: ", len(y_pred)


score_test = f1_score(y_val, y_pred)
"""




