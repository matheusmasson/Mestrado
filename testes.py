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
NEW_USERS2 = '/media/matheus/Files/DataSet/newusers2.pkl'

d = h5py.File(DATASET_PATH, 'r')
dfs = h5py.File(DATASET_FPATH, 'r')

print "*TREINO DA FUNCAO OBJETIVO 2 COM CRIACAO DE USUARIOS"

print "Filtragem do dataset para treino"
song_factors = dfs['factors'][:]
# Get cold index
index_cold = np.where(np.all(song_factors == 0, axis=1))[0]
ds_size = len(song_factors)
list_index = range(0, ds_size)
list_index_not_cold = list(np.delete(np.array(list_index), index_cold))
del song_factors, index_cold, ds_size,  list_index
gc.collect()

all_ids = dfs['id'][list_index_not_cold]

wmf_file = np.load(WMF_FEATURES)
user_vectors = wmf_file['U']
song_vectors = wmf_file['V']
shape_songs_wmf = song_vectors.shape
del song_vectors

print "criacao da matriz para fatoracao"
rows_triples = db.select_all_triples_full()
df = pd.DataFrame.from_records(rows_triples, columns=['userid', 'musicid', 'count', 'train', 'test', 'cold'])
df.columns = ['user', 'music', 'count', 'train', 'test', 'cold']
train_set = df[df['train'] == 1]


idsbanco = list(train_set['music'].unique())
idsbanco = sorted(idsbanco)
idsbanco = list(idsbanco)

print "A"

ids1 = set(idsbanco)
ids2 = set(all_ids)

cf = [i for i in ids1 if i not in ids2]
cf2 = [i for i in ids2 if i not in ids1]


print "comparacao de ids nao cold"
print "pelo banco: "
print cf
print
print "pelo filtro numpy no dataset: "
print cf2






"""
train_set = train_set.sort(['user'])
rowssparse = train_set.to_records(index=False)
dbsparse = np.fromiter(rowssparse, dtype=[('user', int), ('music', int), ('count', int)])
sparse_matrix = sparse.csr_matrix((dbsparse['count'], (dbsparse['user'], dbsparse['music'])))








del rows_triples, df, train_set, rowssparse, dbsparse
gc.collect()

# usuarios iniciais do treino
user_matrix_cnn = user_vectors

print "criacao do target para funcao objetivo 2 (contagem de consumo por musica)"
num_examples = all_ids.shape[0]
newy = np.zeros(num_examples)
rows_counts = db.select_all_rtings_train_full()
dic_ratings = {}
for rw in rows_counts:
    dic_ratings[rw[0]] = rw[1]
iter = 0
for val in all_ids:
    newy[iter] = dic_ratings[val]
    iter += 1
y = newy


y = np.array(y).astype(np.float32)

num_examples = y.shape[0]
print "tamanho do treino: ", num_examples
batchsize = num_examples // 5


def get_batch(bnum):
    ini = bnum * batchsize
    end = ini + batchsize

    y_batch = y[ini:end]
    ids_batch = all_ids[ini:end]

    rg_not_cold = list_index_not_cold[ini:end]

    x_batch = d["spectrograms"][rg_not_cold]
    x_batch = x_batch[:, :, 0:256]
    size = x_batch.shape[0]
    x_batch = x_batch.reshape(size, 1, 256, 256)

    return x_batch, y_batch, ids_batch


songs_matrix_cnn = np.zeros(shape_songs_wmf)

for j in xrange(10):
    for i in xrange((num_examples + batchsize - 1) // batchsize):
        xb, yb, idsb = get_batch(i)
        net1.fit(xb, yb)

        s_pred = net1.predict(xb)
        s_pred = np.vstack(s_pred)

        for inx, val in enumerate(idsb):
            if inx < len(s_pred):
                songs_matrix_cnn[val] = s_pred[inx]
        del xb, yb, idsb, s_pred
        gc.collect()

    if j > 18:
        net1.update_learning_rate = 0.001
    if j > 20:
        net1.update_learning_rate = 0.0001
    gc.collect()



    if (j > 0) and (j % 2 == 0):
        users = wmf.factorize_only_users(songs_matrix_cnn, 1, sparse_matrix)

        del net1.objective_users, net1.train_iter_, net1.eval_iter_, net1.predict_iter_
        gc.collect()

        net1.objective_users = users
        user_matrix_cnn[:] = users
        y_tensor_type = T.TensorType(
                    theano.config.floatX, (False, False))

        iter_funcs = net1._create_iter_funcs(
            net1.layers_, objective_ratings, net1.update,
            y_tensor_type,
            )

        net1.train_iter_, net1.eval_iter_, net1.predict_iter_ = iter_funcs



net1.save_params_to('/media/matheus/Files/aprendizado/conv_full2.pkl')
file = plot_loss(net1)
file.savefig('/media/matheus/Files/aprendizado/conv_full2.png')

pickle.dump(user_matrix_cnn, open(NEW_USERS2, 'wb'))

"""