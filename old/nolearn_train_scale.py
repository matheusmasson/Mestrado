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
from lasagne.layers import get_output, Upscale2DLayer, ReshapeLayer, get_all_params
from nolearn.lasagne import NeuralNet, BatchIterator, TrainSplit
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

import networks as nt
import DataBase as db
import pandas as pd
import wmf
import time
from scipy.ndimage import convolve

warnings.filterwarnings('ignore', module='lasagne')

DATASET_PATH = "/media/matheus/Files/DataSet/spectrograms_dataset_micro.h5"
DATASET_FPATH = "/media/matheus/Files/DataSet/factors_small.h5"
NEW_USERS2 = '/media/matheus/Files/DataSet/newusers2.pkl'


WMF_FEATURES = '/media/matheus/Files/DataSet/testesmall.npz'


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
dynamic_scalerobj = preprocessing.MinMaxScaler(feature_range=(-1, 1))
scale = dynamic_scalerobj.fit_transform(all_factors)

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
user_matrix_cnn2 = np.zeros((300, 50))


CHANGE_RATE = True
iteration_batch_num = []
iteration_batch_num.append(0)
epochatual = []
epochatual.append(0)
inc = [0]

dynamic_scalerW = preprocessing.MinMaxScaler(feature_range=(-5, 5))
dynamic_scalerT = preprocessing.MinMaxScaler(feature_range=(-1, 1))


num_examples = data_train.shape[0]

newy = np.zeros(num_examples)

rows_counts = db.select_all_rtings_train()

dic_ratings = {}





ite = 0
for rw in rows_counts:
    dic_ratings[rw[0]] = rw[1]


print "calculo:"
iter = 0
for val in all_ids:
    newy[iter] = dic_ratings[val]
    iter += 1

print "13 teste"


def on_training_started(net, hist):
    print ""
    #net.load_params_from(PARAMS2)


Xp = data_train[:, :, 0:128]
num_examplesp = len(Xp)

Xp = np.array(Xp).astype(np.float32)
Xp = Xp.reshape(num_examplesp, 1, 128, 128)


def rating_objective(predicted, actual, users, makescale, sumall, scalemin, scales):

    original_ratings = T.sum(actual)

    matrix_user_cnn = users.astype(np.float32)

    pred_scale = predicted

    pred_rat = T.dot(matrix_user_cnn, pred_scale.T)

    if sumall:
        pred_rat = pred_rat[(pred_rat > 0).nonzero()]

    preds = T.sum(pred_rat)

    loss = (original_ratings - preds)**2
    #loss = T.sqrt(loss)

    return T.sum(loss)

def rating_objective_scale(predicted, actual, users, makescale, sumall, scalemin, scales):

    original_ratings = T.sum(actual)

    matrix_user_cnn = users.astype(np.float32)

    pred_scale = predicted
    if makescale:
        pred_scale -= scalemin
        pred_scale /= scales

    pred_rat = T.dot(matrix_user_cnn, pred_scale.T)

    if sumall:
        pred_rat = pred_rat[(pred_rat > 0).nonzero()]

    preds = T.sum(pred_rat)

    loss = (original_ratings - preds)**2
    #loss = T.sqrt(loss)

    return T.sum(loss)


def objective_ratings(layers, loss_function, target, users, makescale, sumall, aggregate=aggregate, deterministic=False, get_output_kw=None):

    if get_output_kw is None:
        get_output_kw = {}

    minscale = 0
    scale = 0

    if deterministic:
        print "usuario 1: "
        print users[1]
        print "usuario 2: "
        print users[100]

    if makescale:
        #dynamic_scalerobj.fit_transform(users)
        minscale = dynamic_scalerobj.min_
        scale = dynamic_scalerobj.scale_

    output_layer = layers[-1]
    network_output = get_output(
        output_layer, deterministic=deterministic, **get_output_kw)
    loss = loss_function(network_output, target, users, makescale, sumall, minscale, scale).sum()

    loss += regularization.regularize_layer_params(
            layers.values(), regularization.l1) * 0.001
    loss += regularization.regularize_layer_params(
            layers.values(), regularization.l2) * 0.002


    loss = loss * 0.0000000001

    return loss


def on_epoch_finish_callback_scale(net, hist):

    epc_atual = len(hist)

    fatora = False

    # TRINA OS USUARIOS DE ACORDO COM AS MUSICAS PREDITAS ENTRE AS EPOCAS
    if epc_atual > 3:
        fatora = True
    else:
        fatora = False

    if epc_atual > 60:
        if epc_atual % 3 == 0:
            fatora = True
        else:
            fatora = False

    if fatora:
        if epc_atual > 30:
            net.update_learning_rate = 0.001

        s_pred = net.predict(Xp)
        s_pred = np.vstack(s_pred)

        songs_matrix_cnn = np.zeros(song_vectors.shape)
        print "criou usuarios"
        print "qtd exemplos: ", len(Xp)
        print "musica 1:"

        #s_pred = dynamic_scalerobj.inverse_transform(s_pred)


        for inx, val in enumerate(all_ids):
            if inx < len(s_pred):
                songs_matrix_cnn[val] = s_pred[inx]

        print songs_matrix_cnn[2]

        users, ignored = wmf.factorize_only_users(songs_matrix_cnn, 1, sparse_matrix)
        net.objective_users = users
        user_matrix_cnn[:] = users

        y_tensor_type = T.TensorType(
                    theano.config.floatX, (False, False))

        iter_funcs = net._create_iter_funcs(
            net.layers_, objective_ratings, net.update,
            y_tensor_type,
            )

        net.train_iter_, net.eval_iter_, net.predict_iter_ = iter_funcs


def nudge_dataset(X, Y):
    """
    This produces a dataset 5 times bigger than the original one,
    by moving the 8x8 images in X around by 1px to left, right, down, up
    """
    direction_vectors = [
        [[0, 1, 0],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [1, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 1],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 1, 0]]]

    shift = lambda x, w: convolve(x.reshape((128, 128)), mode='constant',
                                  weights=w).ravel()
    X = np.concatenate([X] +
                       [np.apply_along_axis(shift, 1, X, vector)
                        for vector in direction_vectors])
    Y = np.concatenate([Y for _ in range(5)], axis=0)
    return X, Y

X = data_train[:, :, 0:128]
y = newy

teste = X.reshape(num_examples, 16384)

X, y = nudge_dataset(teste, y)

num_examples = len(X)
print "quantidade de exemplos:", num_examples

total = num_examples
indices = np.arange(total)
np.random.shuffle(indices)


X = X[indices]
y = y[indices]


X = np.array(X).astype(np.float32)
y = np.array(y).astype(np.float32)
X = X.reshape(num_examples, 1, 128, 128)

la = nt.get_network(128, 128, 1)

"""
mean_users = np.mean(user_matrix_cnn, axis=0)

u2 = user_matrix_cnn[:]
scale = dynamic_scalerobj.fit_transform(u2)


for i in range(300):
    seed = np.random.rand(50) * 0.01
    user_matrix_cnn2[i] = mean_users + seed
"""


net3 = NeuralNet(
    layers=la,
    max_epochs=9,

    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,
    on_training_started=[on_training_started],
    on_epoch_finished=[on_epoch_finish_callback_scale],
    objective_loss_function=rating_objective,
    objective=objective_ratings,
    objective_users=user_matrix_cnn,
    objective_makescale=False,
    objective_sumall=True,
    train_split=TrainSplit(eval_size=0.1),
    batch_iterator_train=BatchIterator(batch_size=128, shuffle=True),

    regression=True,

    verbose=3,
)



net3.fit(X, y)

net3.save_params_to('/media/matheus/Files/aprendizado/conv_new_content1.pkl')
file = plot_loss(net3)
file.savefig('/media/matheus/Files/aprendizado/conv_new_content1.png')

pickle.dump(user_matrix_cnn, open(NEW_USERS2, 'wb'))


print "-----------------"
