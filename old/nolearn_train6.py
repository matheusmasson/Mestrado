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
from nolearn.lasagne import NeuralNet, BatchIterator
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
from scipy.ndimage import convolve

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
user_matrix_cnn2 = np.zeros((300, 50))


CHANGE_RATE = True
iteration_batch_num = []
iteration_batch_num.append(0)
epochatual = []
epochatual.append(0)
inc = [0]

dynamic_scalerW = preprocessing.MinMaxScaler(feature_range=(-5, 5))
dynamic_scalerT = preprocessing.MinMaxScaler(feature_range=(-1, 1))

dynamic_scalerobj = preprocessing.MinMaxScaler(feature_range=(-1, 1))

num_examples = data_train.shape[0]

newy = np.zeros(num_examples)

rows_counts = db.select_all_rtings_train()

dic_ratings = {}

scale = dynamic_scalerobj.fit_transform(user_matrix_cnn)


ite = 0
for rw in rows_counts:
    dic_ratings[rw[0]] = rw[1]

iter = 0
for val in all_ids:
    #newy[iter] = dic_ratings[val]
    newy[iter] = val
    iter += 1


def on_training_started(net, hist):
    print ""
    #net.load_params_from(PARAMS2)


Xp = data_train[:, :, 0:128]
num_examplesp = len(Xp)

Xp = np.array(Xp).astype(np.float32)
Xp = Xp.reshape(num_examplesp, 1, 128, 128)

# gera os indices
total = num_examples * 5
indices = np.arange(total)
np.random.shuffle(indices)

bat_train_atual = []
bat_eval_atual = []

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





la = nt.get_network(128, 128, 1)

bat_train_atual.append(0)
bat_eval_atual.append(0)

X = X[indices]
y = y[indices]

X = np.array(X).astype(np.float32)
y = np.array(y).astype(np.float32)
X = X.reshape(num_examples, 1, 128, 128)


trainsize = num_examples * 0.8
idcs_songs_train = y[0:trainsize]
idcs_songs_valid = y[trainsize:num_examples]


def _sldict(arr, sl):
    if isinstance(arr, dict):
        return {k: v[sl] for k, v in arr.items()}
    else:
        return arr[sl]

class MyTrainSplit(object):
    def __init__(self, eval_size, stratify=True):
        self.eval_size = eval_size
        self.stratify = stratify

    def __call__(self, X, y, net):
        if self.eval_size:

            tam = y.shape[0]
            indx = np.arange(tam)
            trainsize = tam * 0.8
            idcs_t = indx[0: trainsize]
            idcs_e = indices[trainsize: tam]

            X_train, y_train = _sldict(X, idcs_t), y[idcs_t]
            X_valid, y_valid = _sldict(X, idcs_e), y[idcs_e]
        else:
            X_train, y_train = X, y
            X_valid, y_valid = _sldict(X, slice(len(y), None)), y[len(y):]

        return X_train, X_valid, y_train, y_valid


def get_rating_loss(predicted, actual, users, sparse, makescale, sumall, scalemin, scales, battrain, batval, validacao):

    tamanho_batch_predito = predicted.shape[0]
    ids_songs = []

    if not validacao:
        batchatual = battrain[0]
        init = batchatual * 128
        end = init + 128
        ids_songs[:] = idcs_songs_train[init:end]
        battrain[0] += 1
    else:
        batchatual = batval[0]
        init = batchatual * 128
        end = init + 128
        ids_songs[:] = idcs_songs_valid[init:end]
        batval[0] += 1

    loss = []

    for it in range(128):
        if T.gt(tamanho_batch_predito, it):
            idsong = int(ids_songs[it])

            songpandas = sparse[sparse['music'] == idsong]

            usrs_ids = list(songpandas['user'].unique())
            originals = list(songpandas['count'])
            originals = np.array(originals).sum()

            qtdusr = len(usrs_ids)

            newus_matrix = np.zeros((qtdusr, 50))

            for j in range(qtdusr):
                idu = usrs_ids[j]
                newus_matrix[j] = users[idu]

            pred_facts = predicted[it]
            if makescale:
                pred_facts -= scalemin
                pred_facts /= scales

            all_rat_cnn_song = T.dot(newus_matrix, pred_facts.T)
            if sumall:
                all_rat_cnn_song = all_rat_cnn_song[(all_rat_cnn_song > 0).nonzero()]

            predic_rat = T.sum(all_rat_cnn_song)

            loss_song = (originals - predic_rat)**2
            loss.append(loss_song)

    ret = T.sum(loss) + T.sum(actual)
    result = ret - T.sum(actual)

    return result


def objective_ratings(layers, loss_function, target, users, sparse, makescale, sumall, battrain, batval, aggregate=aggregate, deterministic=False, get_output_kw=None):

    if get_output_kw is None:
        get_output_kw = {}

    minscale = 0
    scale = 0

    #print "usuario 1: "
    #print users[1]
    #print "usuario 2: "
    #print users[100]

    if makescale:
        #dynamic_scalerobj.fit_transform(users)
        minscale = dynamic_scalerobj.min_
        scale = dynamic_scalerobj.scale_

    output_layer = layers[-1]
    network_output = get_output(
        output_layer, deterministic=deterministic, **get_output_kw)

    validacao = deterministic

    loss = loss_function(network_output, target, users, sparse, makescale, sumall, minscale, scale, battrain, batval, validacao)

    loss += regularization.regularize_layer_params(
            layers.values(), regularization.l1) * 0.0001
    loss += regularization.regularize_layer_params(
            layers.values(), regularization.l2) * 0.0005

    return loss


def on_epoch_finish_callback_scale(net, hist):

    epc_atual = len(hist)

    print "total de batchs treino: ", bat_train_atual[0]
    print "total de batchs valid: ", bat_eval_atual[0]

    bat_train_atual[0] = 0
    bat_eval_atual[0] = 0

    net.objective_battrain=bat_train_atual,
    net.objective_batval=bat_eval_atual,

    if epc_atual > 4:

        if epc_atual > 30:
            net.update_learning_rate = 0.001

        s_pred = net.predict(Xp)
        s_pred = np.vstack(s_pred)

        songs_matrix_cnn = np.zeros(song_vectors.shape)
        print "criou usuarios"
        print "qtd exemplos: ", len(Xp)
        print "musica 1:"


        for inx, val in enumerate(all_ids):
            if inx < len(s_pred):
                songs_matrix_cnn[val] = s_pred[inx]

        temp = dynamic_scalerobj.inverse_transform(songs_matrix_cnn)
        songs_matrix_cnn = temp[:]
        print songs_matrix_cnn[2]

        users, ignored = testes.wmf.factorize_only_users(songs_matrix_cnn, 1, sparse_matrix)
        net.objective_users = users
        user_matrix_cnn[:] = users

        y_tensor_type = T.TensorType(
                    theano.config.floatX, (False, False))

        iter_funcs = net._create_iter_funcs(
            net.layers_, objective_ratings, net.update,
            y_tensor_type,
            )

        net.train_iter_, net.eval_iter_, net.predict_iter_ = iter_funcs




"""
mean_users = np.mean(user_matrix_cnn, axis=0)

u2 = user_matrix_cnn[:]
scale = dynamic_scalerobj.fit_transform(u2)


for i in range(300):
    seed = np.random.rand(50) * 0.01
    user_matrix_cnn2[i] = mean_users + seed
"""

print "1-- - teste 1: "


net3 = NeuralNet(
    layers=la,
    max_epochs=35,

    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.975,
    on_training_started=[on_training_started],
    on_epoch_finished=[on_epoch_finish_callback_scale],
    objective_loss_function=get_rating_loss,
    objective=objective_ratings,
    objective_users=user_matrix_cnn,
    objective_sparse=train_set,
    objective_makescale=True,
    objective_sumall=True,
    #y_tensor_type=T.ivector,
    train_split=MyTrainSplit(eval_size=0.2),
    #batch_iterator_train=BatchIterator(batch_size=128, shuffle=True),
    objective_battrain=bat_train_atual,
    objective_batval=bat_eval_atual,

    regression=True,

    verbose=3,
)

net3.fit(X, y)

net3.save_params_to('/media/matheus/Files/aprendizado/conv_new_content1.pkl')
file = plot_loss(net3)
file.savefig('/media/matheus/Files/aprendizado/conv_new_content1.png')

pickle.dump(user_matrix_cnn, open(NEW_USERS2, 'wb'))


print "-----------------"
print "terminou teste 1"


