import os, sys, urllib, gzip

import matplotlib

matplotlib.use('Agg')
matplotlib.rcParams['figure.max_open_warning'] = 1000

try:
    import cPickle as pickle
except:
    import pickle
sys.setrecursionlimit(10000)

from lasagne.updates import nesterov_momentum
from lasagne import regularization
from lasagne.layers import get_output, Upscale2DLayer, ReshapeLayer
from nolearn.lasagne import NeuralNet
from nolearn.lasagne.visualize import plot_loss
from lasagne.objectives import aggregate
import numpy as np
import theano
import theano.tensor as T
import h5py
from scipy import sparse
from sklearn import preprocessing
from datetime import datetime, timedelta
import warnings

import testes.networks as nt
import testes.DataBase as db
import pandas as pd
import testes.wmf
import time

warnings.filterwarnings('ignore', module='lasagne')

DATASET_PATH = "/media/matheus/Files/DataSet/spectrograms_dataset_micro.h5"
DATASET_FPATH = "/media/matheus/Files/DataSet/factors_small.h5"
NEW_USERS2 = '/media/matheus/Files/DataSet/newusers2.pkl'


WMF_FEATURES = '/media/matheus/Files/DataSet/testesmall.npz'
NEW_USERS = '/media/matheus/Files/DataSet/newusers_test.pkl'

PARAMS = '/media/matheus/Files/aprendizado/tempparms.pkl'
PARAMS2 = '/media/matheus/Files/aprendizado/conv_test_net_cnn1_w.pkl'


NUM_FACTORS = 50
CHUNK_SIZE = 2048
NUM_CHUNKS = 30
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
print "leu todos"
data_eval = data_train[min_eval:max_eval]
#data_eval = data_train[900:1028]
print "copiou os de validacao"

print "vai ler os fatores"
# Get the factors and scale all the values between 0 and 1 for convnet performance
all_factors = dfs['factors'][:]
all_factors = all_factors[list_index_not_cold]
all_ids = dfs['id'][list_index_not_cold]
all_f_str = dfs['fstr'][:]
all_f_str = dfs['fstr'][list_index_not_cold]
print "leu todos os fatores"


min_max_scaler = preprocessing.MinMaxScaler()
all_factors = min_max_scaler.fit_transform(all_factors)

labels_train = all_factors[:]
labels_eval = all_factors[min_eval:max_eval]

wmf_file = np.load(WMF_FEATURES)
user_vectors = wmf_file['U']
song_vectors = wmf_file['V']

#new_users = pickle.load(open(NEW_USERS, 'rb'))
new_users = []
print "leu novos usuarios"
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

user_matrix_cnn = user_vectors
songs_matrix_cnn = np.zeros(song_vectors.shape)


CHANGE_RATE = True
iteration_batch_num = []
iteration_batch_num.append(0)
epochatual = []
epochatual.append(0)
inc = [0]

net_runs = [3]
NUM_UNSUPERVISED = 1

dynamic_scalerW = preprocessing.MinMaxScaler(feature_range=(-5, 5))
dynamic_scalerT = preprocessing.MinMaxScaler(feature_range=(-1, 1))

dynamic_scalerobj = preprocessing.MinMaxScaler(feature_range=(-1, 1))

scale = dynamic_scalerobj.fit_transform(user_vectors)

first = []
first.append(0)
num_examples = data_train.shape[0]

newy = np.zeros(num_examples)

iter = 0
for val in all_ids:
    r = list(train_set[train_set['music'] == val]['count'])

    total = np.array(r).sum()

    newy[iter] = total
    iter += 1

print "criou as labels"


def on_training_started(net, hist):

    iteration_batch_num[0] = 0
    if first[0] == 0:
        print "carga dos parametros 1"
        net.load_params_from(PARAMS2)
    else:
        print "carga dos parametros 2"
        net.load_params_from(PARAMS)

    first[0] = 1


def on_epoch_finish_callback2(net, hist):
    iteration_batch_num[0] = 0

def none_loss(predicted, actual):
    print "loss"

    return (actual - predicted) - (actual - predicted)


def rating_objective(predicted, actual):

    original_ratings = T.sum(actual)

    matrix_user_cnn = user_matrix_cnn.astype(np.float32)

    pred_scale = predicted
    pred_scale -= dynamic_scalerobj.min_
    pred_scale /= dynamic_scalerobj.scale_

    all_ratings_cnn = T.dot(matrix_user_cnn, pred_scale.T)

    pred_rat = all_ratings_cnn[(all_ratings_cnn > 0).nonzero()]
    preds = T.sum(pred_rat)

    loss = (original_ratings - preds)**2
    loss = T.sqrt(loss)

    return T.sum(loss)


if 3 in net_runs:

    print "3- hybrid objective 1818!!"
    X = data_train[:, :, 0:128]
    y = newy

    num_examples = data_train.shape[0]

    X = np.array(X).astype(np.float32)
    y = np.array(y).astype(np.float32)
    X = X.reshape(num_examples, 1, 128, 128)

    layers = nt.get_network(128, 128, 1)

    epcs = 2

    for i in range(40):
        print "treino:", str(i + 1)

        if i == 0:
            epcs = 10
        else:
            epcs = 2
        if i > 6:
            epcs = 3

        if i > 18:
            epcs = 4
        if i > 32:
            epcs = 10

        net3 = NeuralNet(
            layers=layers,
            max_epochs=epcs,

            update=nesterov_momentum,
            update_learning_rate=0.01,
            update_momentum=0.975,
            on_training_started=[on_training_started],
            on_epoch_finished=[on_epoch_finish_callback2],
            objective_loss_function=rating_objective,

            regression=True,

            verbose=3,
        )


        net3.fit(X, y)

        s_pred = net3.predict(X)
        s_pred = np.vstack(s_pred)

        net3.save_params_to(PARAMS)
        time.sleep(1)
        del net3

        print "vetor 1 pos cnn"
        print s_pred[0]

        for inx, val in enumerate(all_ids):
            if inx < len(s_pred):
                songs_matrix_cnn[val] = s_pred[inx]

        print "run wmf 2 epochs, u - > uxv"

        temp = dynamic_scalerobj.inverse_transform(songs_matrix_cnn)
        songs_matrix_cnn = temp[:]

        user_matrix_cnn, ignored = testes.wmf.factorize_only_users(songs_matrix_cnn, 1, sparse_matrix)

        scale = dynamic_scalerobj.fit_transform(user_matrix_cnn)


    print "ultimo treino"
    net5 = NeuralNet(
        layers=layers,
        max_epochs=10,
        on_training_started=[on_training_started],
        on_epoch_finished=[on_epoch_finish_callback2],


        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.975,
        objective_loss_function=rating_objective,

        regression=True,

        verbose=3,
    )

    net5.fit(X, y)

    net5.save_params_to('/media/matheus/Files/aprendizado/conv_hybrid1.pkl')
    file = plot_loss(net5)
    file.savefig('/media/matheus/Files/aprendizado/conv_hybrid1.png')

    pickle.dump(user_matrix_cnn, open(NEW_USERS2, 'wb'))
