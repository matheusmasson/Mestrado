import os, sys, urllib, gzip

import matplotlib

matplotlib.use('Agg')
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
import wmf
import networks as nt
import DataBase as db
import metrics
import time
from scipy import sparse
import pandas as pd


warnings.filterwarnings('ignore', module='lasagne')

DATASET_PATH = "/media/matheus/Files/DataSet/spectrograms_dataset_micro3.h5"
DATASET_FPATH = "/media/matheus/Files/DataSet/factors_small.h5"

NETW_PARAMS = '/media/matheus/Files/aprendizado/teste_obj2_somusicas.pkl'

NEW_USERS = '/media/matheus/Files/DataSet/newusers.pkl'
WMF_FEATURES = '/media/matheus/Files/DataSet/testesmall.npz'



dataset = h5py.File(DATASET_PATH, 'r')
dfs = h5py.File(DATASET_FPATH, 'r')

rowsids = db.select_test_cold_song_ids_small()

ids_test_cold = []
for rw in rowsids:
    ids_test_cold.append(int(rw[0]))

init_size = len(dfs['factors'][:])
real_size = len(ids_test_cold)


s1 = dfs['factors'][:].shape[1]
s3 = dataset['spectrograms'][:].shape[1]
s4 = dataset['spectrograms'][:].shape[2]


song_factors = np.zeros((real_size, s1))
song_ids = []
data_test = np.zeros((real_size, s3, s4))

indreal = 0
for i in range(0, init_size):
    if int(dataset['id'][i]) in ids_test_cold:
        song_factors[indreal] = dfs['factors'][i]
        data_test[indreal] = dataset["spectrograms"][i]
        song_ids.append(int(dataset['id'][i]))
        indreal += 1

print "tamanho teste:", len(song_ids)


# Create empty labels for prediction
labels_test = np.zeros(song_factors.shape)

# get spectrograms matrix shape
data_test_shape = data_test.shape

dataset.close()
#del song_factors

print "Compute Predictions"

num_examples_test = data_test_shape[0]
print "Numero de exemplos de teste: ", num_examples_test

print "Load model parameters"

print "formato matriz teste: ", data_test.shape
print "formato ajustado:"

X = data_test[:, :, 0:256]
print "*formato ajustado rede 1:", X.shape

layers = nt.get_network(256, 256, 4)
X = np.array(X).astype(np.float32)
X = X.reshape(num_examples_test, 1, 256, 256)

net = NeuralNet(
    layers=layers,

    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.975,
    regression=True,

    verbose=3,
)


net.load_params_from(NETW_PARAMS)

y_pred = net.predict(X)

print "formato predito: ", y_pred.shape

all_predictions = np.vstack(y_pred)

print "formato predito ajustado:", all_predictions.shape


print "Number of predictions: ", len(all_predictions)






"""


print "CARREGA USUARIOS!!!"
dynamic_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
#temp = dynamic_scaler.fit_transform(all_predictions)

userscnn = pickle.load(open('/media/matheus/Files/DataSet/newusers2.pkl', 'rb'))



print "---TESTE DA METRICA COM USUARIOS MEDIOS!!"
wmf_file = np.load(WMF_FEATURES)
songswmf = np.array(wmf_file['V'])
temp = dynamic_scaler.fit_transform(songswmf)

temp2 = dynamic_scaler.inverse_transform(all_predictions)
all_predictions = temp2[:]

training_user_facts = np.array(wmf_file['U'])
users = np.array(training_user_facts)

u2 = np.zeros(users.shape)
mean_users = np.mean(users, axis=0)

u2 = np.zeros(users.shape)
u3 = np.zeros(users.shape)
for i in range(users.shape[0]):
    seed = np.random.rand(50) * 0.0001
    u2[i] = mean_users + seed
    u3[i] = mean_users



metrics.generate_metrics_convnet(all_predictions, song_ids, u3, True)

"""


print "2nova rede.."

#metrics.generate_metrics_convnet(all_predictions, song_ids, userscnn, True)






# Scale predictions to wmf scales (scale back from before train) to compare results with wmf results

# First do the same scale done in train to get the same scale range

# Load dataset from file

dataset = h5py.File(DATASET_FPATH, 'r')

song_factors = dataset['factors'][:]

# Get cold index
index_cold = np.where(np.all(song_factors == 0, axis=1))[0]
ds_size = len(song_factors)
list_index = range(0, ds_size)
list_index_not_cold = list(np.delete(np.array(list_index), index_cold))
song_factors = song_factors[list_index_not_cold]

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
song_factors = min_max_scaler.fit_transform(song_factors)

# INVERTE A ESCALA DA PREDICAO PELA ESCALA DA WMF
#all_predictions = min_max_scaler.inverse_transform(all_predictions)

wmf_file = np.load(WMF_FEATURES)
training_user_facts = wmf_file['U']


#min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
#training_user_facts = min_max_scaler.fit_transform(training_user_facts)




print "18.2 teste"

del data_test

#userscnn = pickle.load(open('/media/matheus/Files/DataSet/newusers2.pkl', 'rb'))
#metrics.generate_metrics_convnet(all_predictions, song_ids, userscnn, True)
metrics.generate_metrics_convnet(all_predictions, song_ids, None, False)


print "Criando matriz de treino"

rows_triples = db.select_all_triples_small()
df = pd.DataFrame.from_records(rows_triples, columns=['userid', 'musicid', 'count', 'train', 'test', 'cold'])
df.columns = ['user', 'music', 'count', 'train', 'test', 'cold']
train_set = df[df['train'] == 1]
train_set = train_set.sort(['user'])
rowssparse = train_set.to_records(index=False)
# Load Matrix from DataBase
dbsparse = np.fromiter(rowssparse, dtype=[('user', int), ('music', int), ('count', int)])
# Create sparse matrix
sparse_matrix = sparse.csr_matrix((dbsparse['count'], (dbsparse['user'], dbsparse['music'])))

print "Carregando vetores latentes iniciais 18"

wmf_file = np.load(WMF_FEATURES)
training_user_facts = np.zeros(wmf_file['U'].shape)
training_song_facts = wmf_file['V']

predicted_factors_convnet_test = np.zeros(training_song_facts.shape)
for inx, val in enumerate(song_ids):
        if inx < len(all_predictions):
            if not np.all(training_song_facts[val] == 0):
                predicted_factors_convnet_test[val] = all_predictions[inx]

U, C = wmf.factorize_only_users(predicted_factors_convnet_test, 1, sparse_matrix, 50)
#pickle.dump(U, open(NEW_USERS, 'wb'))
#print "salvou usuarios .."

print "teste modelo com usuarios criados"
metrics.generate_metrics_convnet(all_predictions, song_ids, U, True)








