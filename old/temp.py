import numpy as np
import testes.DataBase as db
import pandas as pd
import h5py
from sklearn import preprocessing
from scipy import sparse

try:
    import cPickle as pickle
except:
    import pickle

DATASET_PATH = "/media/matheus/Files/DataSet/spectrograms_dataset_micro.h5"
DATASET_FPATH = "/media/matheus/Files/DataSet/factors_small.h5"

WMF_FEATURES = '/media/matheus/Files/DataSet/modelwmf.npz'
NEW_USERS = '/media/matheus/Files/DataSet/newusers_test.pkl'


d = h5py.File(DATASET_PATH, 'r')
dfs = h5py.File(DATASET_FPATH, 'r')

max = 9000
min_train = 0
min_eval = 8231
max_eval = 8990
NUM_FACTORS = 50


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



rows_triples = db.select_all_triples_small()

df = pd.DataFrame.from_records(rows_triples, columns=['userid', 'musicid', 'count', 'train', 'test'])

df.columns = ['user', 'music', 'count', 'train', 'test']
train_set = df[df['train'] == 1]


new_users = pickle.load(open(NEW_USERS, 'rb'))


exemplo = min_max_scaler.inverse_transform(labels_train[0])

exemplo2 = min_max_scaler.inverse_transform(labels_train[1])

if np.array_str(exemplo, precision=5) in all_f_str:
    idx_song = np.where(all_f_str == np.array_str(exemplo, precision=5))[0][0]
    idx_song = all_ids[idx_song]

    song_ratings = train_set[train_set['music'] == idx_song]

    all_losses = []

    for index, row in song_ratings.iterrows():
        original_rate = row['count']

        usr_id = row['user']

        usr_fact_wmf = user_vectors[usr_id]
        usr_fact_cnn = new_users[usr_id]

        wmf_rate = usr_fact_wmf.dot(exemplo.T)
        pred_rate = usr_fact_cnn.dot(exemplo2.T)

        loss = (original_rate - pred_rate - wmf_rate)**2
        all_losses.append(loss)

    obj_loss = np.sum(all_losses)
