import os, sys, urllib, gzip

import matplotlib

matplotlib.use('Agg')
matplotlib.rcParams['figure.max_open_warning'] = 1000

try:
    import cPickle as pickle
except:
    import pickle
sys.setrecursionlimit(10000)


import numpy as np

import h5py
from scipy import sparse
from sklearn import preprocessing
from datetime import datetime, timedelta
import warnings

import DataBase as db
import pandas as pd
import wmf
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
"""

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
"""
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
"""
user_matrix_cnn = user_vectors
user_matrix_cnn2 = np.zeros((300, 50))


CHANGE_RATE = True
iteration_batch_num = []
iteration_batch_num.append(0)
epochatual = []
epochatual.append(0)
inc = [0]

#dynamic_scalerW = preprocessing.MinMaxScaler(feature_range=(-5, 5))
#dynamic_scalerT = preprocessing.MinMaxScaler(feature_range=(-1, 1))

#dynamic_scalerobj = preprocessing.MinMaxScaler(feature_range=(-1, 1))

num_examples = data_train.shape[0]

newy = np.zeros(num_examples)

rows_counts = db.select_all_rtings_train()

dic_ratings = {}

#scale = dynamic_scalerobj.fit_transform(song_vectors)




"""

#testes
import metrics as mte

songs = np.array(song_vectors)
users = np.array(user_vectors)



#t1 = np.array(user_vectors)

#md = np.mean(t1, axis=0)
#print md


print "TOTAL COUNTS BANCO: ", 4701424
print

r = users.dot(songs.T)
totalwmf = r.sum()

totalwmffiltrado = r[r > 0]
totalwmffiltrado = totalwmffiltrado.sum()

print
print "total wmf nao filtrado: ", totalwmf
print "total wmf filtrado: ", totalwmffiltrado
print


dynamic_scalerT = preprocessing.MinMaxScaler(feature_range=(-1, 1))
songs = dynamic_scalerT.fit_transform(songs)

users, ignored = wmf.factorize_only_users(songs, 1, sparse_matrix)

r = users.dot(songs.T)
totalwmf = r.sum()

totalwmffiltrado = r[r > 0]
totalwmffiltrado = totalwmffiltrado.sum()

print
print "2-total wmf nao filtrado: ", totalwmf
print "2-total wmf filtrado: ", totalwmffiltrado
print



"""

dynamic_scalerobj.fit_transform(songs)

print "song scale min: ", dynamic_scalerobj.min_
print
print "song scale: ", dynamic_scalerobj.scale_
print

dynamic_scalerobj.fit_transform(users)

print "usr scale min: ", dynamic_scalerobj.min_
print
print "usr scale: ", dynamic_scalerobj.scale_
print

print
print "teste rating com media de usuarios"
mean_users = np.mean(users, axis=0)

u2 = np.zeros(users.shape)
u3 = np.zeros(users.shape)
for i in range(users.shape[0]):
    seed = np.random.rand(50) * 0.0001
    u2[i] = mean_users + seed
    u3[i] = mean_users

print

totmeanrand = u2.dot(songs.T)
totmean = u3.dot(songs.T)

totmeanrand = totmeanrand.sum()
totmean = totmean.sum()

print "total count mean aleatorio: ", totmeanrand
print "total count mean fixo: ", totmean

print
print


test_set = df[df['test'] == 1]
test_set = test_set[list(['user', 'music', 'count'])]
test_songs_ids = list(set(test_set['music']))
test_set = test_set[test_set['music'].isin(test_songs_ids)]






#teste 5
V = wmf.factorize_only_songs(u2, 1, sparse_matrix)
test_all_metrics_wmf = mte.validation(test_set, sparse_matrix, u2, V)

wmf1 = ''
for k in test_all_metrics_wmf:
    wmf1 += 'WMF Metric: ' + str(k) + ' : ' + '{0:.10f}'.format(test_all_metrics_wmf[k])

print "5- wmf com media aleatoria de usuarios e musicas fatoradas - Result: ", 'TEST WMF', " Values: ", wmf1

"""

