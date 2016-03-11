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
dbsparse = np.fromiter(rowssparse, dtype=[('user', int), ('music', int), ('count', int)])

sparse_matrix = sparse.csr_matrix((dbsparse['count'], (dbsparse['user'], dbsparse['music'])))

user_matrix_cnn = user_vectors
user_matrix_cnn2 = np.zeros((250, 50))


CHANGE_RATE = True
iteration_batch_num = []
iteration_batch_num.append(0)
epochatual = []
epochatual.append(0)
inc = [0]

dynamic_scalerW = preprocessing.MinMaxScaler(feature_range=(-5, 5))
dynamic_scalerT = preprocessing.MinMaxScaler(feature_range=(-1, 1))

#dynamic_scalerobj = preprocessing.MinMaxScaler(feature_range=(-1, 1))

#scale = dynamic_scalerobj.fit_transform(user_vectors)

num_examples = data_train.shape[0]

newy = np.zeros((num_examples, 2))

rows_counts = db.select_all_rtings_train()

dic_ratings = {}

ite = 0
for rw in rows_counts:
    dic_ratings[rw[0]] = rw[1]


print "calculo:"
iter = 0
for val in all_ids:
    newy[iter] = [val, dic_ratings[val]]
    iter += 1

print "criou as labels"

X = data_train[:, :, 0:128]
y = newy

X = np.array(X).astype(np.float32)
y = np.array(y).astype(np.float32)
X = X.reshape(num_examples, 1, 128, 128)

la = nt.get_network(128, 128, 1)


def on_training_started(net, hist):
    print ""
    #net.load_params_from(PARAMS2)


def rating_objective(predicted, actual, users, sparse, songids, scalemin, scales):

    qtdbatch = actual.shape[0]

    loss = []

    targ = actual.eval()

    for i in range(128):
        if T.gt(qtdbatch, i):

            id = targ[i][0]
            orig_rat = targ[i][1]

            sp_filt = sparse[np.where(sparse[:, 1] == id)]
            usrs_ids = list(sp_filt[:, 0])
            qtdusr = len(usrs_ids)

            newus_matrix = np.zeros((qtdusr, 50))

            for j in range(qtdusr):
                idu = usrs_ids[j]
                newus_matrix[j] = users[idu]

            pred_facts = predicted[i]

            all_rat_cnn_song = T.dot(newus_matrix, pred_facts.T)
            preds = T.sum(all_rat_cnn_song)

            song_loss = (orig_rat - preds)**2
            loss.append(song_loss)

    result = T.sum(loss)
    return result


def rating_objective2(predicted, actual, users, sparse, songids, scalemin, scales):

    original_ratings = T.sum(actual)

    matrix_user_cnn = users.astype(np.float32)

    pred_scale = predicted
    #pred_scale -= scalemin
    #pred_scale /= scales

    all_ratings_cnn = T.dot(matrix_user_cnn, pred_scale.T)

    #pred_rat = all_ratings_cnn[(all_ratings_cnn > 0).nonzero()]
    preds = T.sum(all_ratings_cnn)

    loss = (original_ratings - preds)**2
    loss = T.sqrt(loss)

    return T.sum(loss)


def objective_ratings(layers, loss_function, target, users, sparse, songids, aggregate=aggregate, deterministic=False, get_output_kw=None):

    if get_output_kw is None:
        get_output_kw = {}

    #dynamic_scalerobj.fit_transform(users)
    #min = dynamic_scalerobj.min_
    #scales = dynamic_scalerobj.scale_

    output_layer = layers[-1]
    network_output = get_output(
        output_layer, deterministic=deterministic, **get_output_kw)
    #loss = rating_objective(network_output, target, users, min, scales).sum()
    loss = loss_function(network_output, target, users, sparse, songids, 0, 0).sum()

    loss += regularization.regularize_layer_params(
            layers.values(), regularization.l1) * 0.001
    loss += regularization.regularize_layer_params(
            layers.values(), regularization.l2) * 0.005


    return loss


def on_epoch_finish_callback2(net, hist):

    epc_atual = len(hist)

    if epc_atual > 0:
        #net.update_learning_rate = 0.001

        s_pred = net.predict(X)
        s_pred = np.vstack(s_pred)

        songs_matrix_cnn = np.zeros(song_vectors.shape)

        for inx, val in enumerate(all_ids):
            if inx < len(s_pred):
                songs_matrix_cnn[val] = s_pred[inx]

        #temp = dynamic_scalerobj.inverse_transform(songs_matrix_cnn)
        #songs_matrix_cnn = temp[:]

        users, ignored = testes.wmf.factorize_only_users(songs_matrix_cnn, 1, sparse_matrix)
        net.objective_loss_function = rating_objective
        net.objective_users = users

        y_tensor_type = T.TensorType(
                    theano.config.floatX, (False, False))

        iter_funcs = net._create_iter_funcs(
            net.layers_, objective_ratings, net.update,
            y_tensor_type,
            )

        net.train_iter_, net.eval_iter_, net.predict_iter_ = iter_funcs



print "3- hybrid objective 18!!"
"""

net0 = NeuralNet(
    layers=la,
    max_epochs=30,

    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.975,
    regression=True,

    verbose=3,
)

net0.load_params_from(PARAMS2)

s_pred = net0.predict(X)
s_pred = np.vstack(s_pred)

songs_matrix_cnn = np.zeros(song_vectors.shape)

for inx, val in enumerate(all_ids):
    if inx < len(s_pred):
        songs_matrix_cnn[val] = s_pred[inx]

#temp = dynamic_scalerobj.inverse_transform(songs_matrix_cnn)
#songs_matrix_cnn = temp[:]

u, ignored = wmf.factorize_only_users(songs_matrix_cnn, 1, sparse_matrix)
user_matrix_cnn = u

print "criou usuarios"
"""

mean_users = np.mean(user_matrix_cnn, axis=0)

for i in range(250):
    user_matrix_cnn2[i] = mean_users

print "us mean"


ids_songs = list(all_ids)

net3 = NeuralNet(
    layers=la,
    max_epochs=25,

    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.975,
    on_training_started=[on_training_started],
    on_epoch_finished=[on_epoch_finish_callback2],
    objective_loss_function=rating_objective2,
    objective=objective_ratings,
    objective_users=user_matrix_cnn2,
    objective_sparse=dbsparse,
    objective_songids=ids_songs,

    regression=True,

    verbose=3,
)

net3.fit(X, y)

net3.save_params_to('/media/matheus/Files/aprendizado/conv_hybrid1.pkl')
file = plot_loss(net3)
file.savefig('/media/matheus/Files/aprendizado/conv_hybrid1.png')

pickle.dump(user_matrix_cnn, open(NEW_USERS2, 'wb'))
