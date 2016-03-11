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

user_matrix_cnn = np.zeros(user_vectors.shape)
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


def on_training_started(net, hist):
    print "carga dos parametros"
    iteration_batch_num[0] = 0
    net.load_params_from(PARAMS)

def on_epoch_finish_callback2(net, hist):
    iteration_batch_num[0] = 0

def none_loss(predicted, actual):
    print "loss"

    return (actual - predicted) - (actual - predicted)


def rating_objective(predicted, actual):
    print "chamou"
    return (predicted - actual)**2


def compute_loss(predicted):
    batch_num = iteration_batch_num[0]
    init = batch_num * 128
    end = init + 128
    music_ids = all_ids[init:end]

    origs = list(train_set[train_set['music'].isin(music_ids)]['count'])
    ids_usrs = list(train_set[train_set['music'].isin(music_ids)]['user'].unique())

    original_ratings = T.sum(origs)
    matrix_user_cnn = np.zeros((len(ids_usrs), 50))

    idxu = 0
    for idus in ids_usrs:
        matrix_user_cnn[idxu] = user_matrix_cnn[idus]
        idxu += 1

    matrix_user_cnn = matrix_user_cnn.astype(np.float32)

    all_ratings_cnn = T.dot(matrix_user_cnn, predicted.T)

    pred_rat = all_ratings_cnn[(all_ratings_cnn > 0).nonzero()]
    preds = T.sum(pred_rat)

    loss = (original_ratings - preds)**2

    iteration_batch_num[0] += 1

    return T.sum(loss)


def objective_ratings(layers,
              loss_function,
              target,
              aggregate=aggregate,
              deterministic=False,
              l1=0.1,
              l2=0.5,
              get_output_kw=None):

    if get_output_kw is None:
        get_output_kw = {}
    output_layer = layers[-1]

    network_output = get_output(
        output_layer, deterministic=deterministic, **get_output_kw)

    loss = aggregate(loss_function(network_output, target))
    rating_loss = compute_loss(network_output)
    loss = (loss * 0)
    loss += rating_loss

    return loss




if 3 in net_runs:

    print "3- hybrid objective 18!!"
    X = data_train[:, :, 0:128]
    y = labels_train

    num_examples = data_train.shape[0]

    layers = nt.get_network(128, 128, 5)
    X = np.array(X).astype(np.float32)
    y = np.array(y).astype(np.float32)
    X = X.reshape(num_examples, 1, 128, 128)

    netpre = NeuralNet(
        layers=layers,
        max_epochs=1,
        regression=True,

        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.975,
        #objective=objective_none,

        objective_loss_function=none_loss,


        verbose=3,
    )

    netpre.fit(X, y)

    s_pred = netpre.predict(X)
    s_pred = np.vstack(s_pred)




    for inx, val in enumerate(all_ids):
        if inx < len(s_pred):
            songs_matrix_cnn[val] = s_pred[inx]


    print "generate first users train 2 epcs"

    # CHECAR!!!
    #V = min_max_scaler.inverse_transform(V)
    temp = dynamic_scalerT.fit_transform(songs_matrix_cnn)
    songs_matrix_cnn = temp[:]

    user_matrix_cnn, ignored = testes.wmf.factorize_only_users(songs_matrix_cnn, 2, sparse_matrix)

    #print "run wmf to update songs 1 epochs"
    #user_matrix_cnn, songs_matrix_cnn = wmf.runwmf(u_pred, songs_matrix_cnn, 1, sparse_matrix)

    #temp = dynamic_scalerT.fit_transform(songs_matrix_cnn)
    #songs_matrix_cnn = temp[:]

    y = np.zeros(labels_train.shape)

    for inx, val in enumerate(all_ids):
        if inx < len(songs_matrix_cnn):
            y[inx] = songs_matrix_cnn[val]

    y = np.array(y).astype(np.float32)
    print "vetor 1 pos fatoracao"
    print y[0]

    netpre.save_params_to(PARAMS)
    time.sleep(1)
    del netpre

    layers = nt.get_network(128, 128, 1)

    epcs = 20

    for i in range(10):
        print "treino:", str(i + 1)

        if i == 0:
            epcs = 40
        elif i == 1:
            epcs = 20
        else:
            epcs += 1


        net3 = NeuralNet(
            layers=layers,
            max_epochs=epcs,

            update=nesterov_momentum,
            update_learning_rate=0.01,
            update_momentum=0.975,
            on_training_started=[on_training_started],
            on_epoch_finished=[on_epoch_finish_callback2],
            objective_loss_function=rating_objective,
            objective=objective_ratings,

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
        #temp = dynamic_scalerW.transform(songs_matrix_cnn)
        #songs_matrix_cnn = temp[:]


        user_matrix_cnn, ignored = testes.wmf.factorize_only_users(songs_matrix_cnn, 4, sparse_matrix)

        #user_matrix_cnn, songs_matrix_cnn = wmf.runwmf(user_matrix_cnn, songs_matrix_cnn, 1, sparse_matrix)

        #temp = dynamic_scalerT.fit_transform(songs_matrix_cnn)
        #songs_matrix_cnn = temp[:]

        y = np.zeros(labels_train.shape)

        for inx, val in enumerate(all_ids):
            if inx < len(songs_matrix_cnn):
                y[inx] = songs_matrix_cnn[val]

        y = np.array(y).astype(np.float32)



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
        objective=objective_ratings,

        regression=True,

        verbose=3,
    )

    net5.fit(X, y)
    s_pred = net5.predict(X)
    s_pred = np.vstack(s_pred)

    temp = dynamic_scalerT.fit_transform(user_matrix_cnn)

    user_matrix_cnn = temp[:]

    net5.save_params_to('/media/matheus/Files/aprendizado/conv_hybrid1.pkl')
    file = plot_loss(net5)
    file.savefig('/media/matheus/Files/aprendizado/conv_hybrid1.png')

    pickle.dump(user_matrix_cnn, open(NEW_USERS2, 'wb'))
