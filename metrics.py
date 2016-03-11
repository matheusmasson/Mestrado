import DataBase as db
import pandas as pd
import numpy as np
from scipy import sparse
from datetime import datetime
from scipy.sparse import csr_matrix
import theano
import theano.tensor as T
import gc
import time
from datetime import datetime

#import analysis

NUM_RECOMENDS = 500
ANALYSE = True

MODEL_PATH = '/media/matheus/Files/DataSet/testefull.npz'
MODEL_PATH_SMALL = '/media/matheus/Files/DataSet/testesmall.npz'

hard_songs = {}


def rr(predicted, true):
    for i, x in enumerate(predicted):
        if x in true:
            return 1.0 / (i + 1)
    return 0


def map2(actual, predicted, k):

    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)
        else:
            if p in hard_songs.keys():
                hard_songs[p] += 1
            else:
                hard_songs[p] = 1

    if not actual:
        return 0.0

    return score / min(len(actual), k)



def prec(predicted, true, k, ignore_missing=False):
    if len(predicted) == 0:
        return 0
    correct = len(set(predicted[:k]).intersection(set(true)))
    num_predicted = k
    if len(predicted) < k and ignore_missing:
        num_predicted = len(predicted)
    return float(correct) / num_predicted

def average_precision(rec_songs, M, k):
    """ This function computes the average precision at each recall point
    - param :
             rec_songs : list of recommended songs
             M : user-song matrix
    """

    np = len(M)
    # print "np:", np
    nc = 0.0
    mapr_user = 0.0
    for j,s in enumerate(rec_songs):
        if j >= k:
            break
        if s in M:
            nc += 1.0
            mapr_user += nc/(j+1)
    mapr_user /= min(np, k)
    return mapr_user


def compute_metrics(recommended, known, num):
    if not known:
        return None
    return {'map': average_precision(recommended, known, num),
            'map2': map2(known, recommended, num),
            'prec@5': prec(recommended, known, 5),
            'prec@10': prec(recommended, known, 10),
            'prec@15': prec(recommended, known, 15),
            'prec@20': prec(recommended, known, 20),
            'mrr': rr(recommended, known)}


def compute_all_metrics(idusrs, recs_songs, num, test_set):
    all_metrics = {'map': 0.0, 'map2': 0.0, 'prec@5': 0.0, 'prec@15': 0.0, 'mrr': 0, 'prec@10': 0.0, 'prec@20': 0.0}
    count = 0

    # Create sparse matrix from test to speed up the select on test songs
    rowssparse = test_set.to_records(index=False)
    dbsparse = np.fromiter(rowssparse, dtype=[('user', int), ('music', int), ('count', int)])
    sparse_matrix = sparse.csr_matrix((dbsparse['count'], (dbsparse['user'], dbsparse['music'])))
    sparse_matrix = sparse_matrix.tocsr()

    for i, id_usr in enumerate(idusrs):

        test_songs = list(sparse_matrix[id_usr].nonzero()[1])

        # print "User: ", id_usr, " Qtd songs: ", usr_songs.__len__(), " Qtd recs: ", np.array(recs_songs[i]).__len__()

        if len(recs_songs[i]) > 0 and len(test_songs) > 0:
            metrics_user = compute_metrics(recs_songs[i], test_songs, num)
            if metrics_user:
                for m, val in metrics_user.iteritems():
                    all_metrics[m] += val
                count += 1

    for m in all_metrics:
        all_metrics[m] /= float(count)
    return all_metrics


def remove_train_scores(scores, train_sparse):
    col = train_sparse.indices

    if isinstance(scores, csr_matrix):
        max_score = scores.data.max()
    else:
        max_score = scores.max()

    data = max_score * np.ones(col.shape)

    # build up the row (user) indices
    # - we can't just use row,col = train.nonzero() as this eliminates
    # u,i for which train[u,i] has been explicitly set to zero

    row = np.zeros(col.shape)
    for u in range(train_sparse.shape[0]):
        start, end = train_sparse.indptr[u], train_sparse.indptr[u + 1]
        if end > start:
            row[start:end] = u
    return scores - csr_matrix((data, (row, col)), shape=scores.shape)


def validation2(test_set, sparse_train, user_factors, song_factors):
    user_vectors = user_factors
    song_vectors = song_factors

    # Select test user ids
    user_ids = list(set(test_set['user']))

    for j, id_us_filter in enumerate(user_ids):
        if int(id_us_filter) >= (user_vectors.shape[0] - 1):
            user_ids.remove(id_us_filter)

    print "Predict ratings"

    # predict ratings for all

    # Select just test users in training data
    user_vectors = np.asfortranarray([user_vectors[index] for index in user_ids])
    sparse_train = sparse_train[user_ids, :]

    recs = []

    # need to process in parts to avoid memory error

    div = len(user_vectors) / 1000

    print "Split validation in ", div, " parts"

    for i in range(0, div):

        print "part: ", i

        start = i * 1000
        end = start + 1000

        # Make sure that goes till the end
        if i == (div - 1):
            end = len(user_vectors)

        uv_part = np.asfortranarray(user_vectors[range(start, end), :])
        ratings = uv_part.dot(song_vectors.T)

        # update score removing already know rating from training
        scores = np.array(remove_train_scores(ratings, sparse_train[start:end]))

        # recommend songs for each user
        count = 0
        for uid in user_ids[start:end]:
            recs.append([i for i in scores[count].argsort()[::-1] if scores[count][i] > 0][:NUM_RECOMENDS])
            count += 1

    met = compute_all_metrics(user_ids, recs, NUM_RECOMENDS, test_set)

    return met



def validation(test_set, user_vectors, song_vectors, train_set, ids_used, clear):
    user_vectors = np.array(user_vectors)
    song_vectors = np.array(song_vectors)

    # Select test user ids
    user_ids = list(test_set['user'].unique())
    size_usrs_test = int(user_vectors.shape[0])

    #TODO REMOVE THIS
    #user_ids = user_ids[0:40000]
    #user_ids.remove(user_ids > size_usrs_test)

    print "Predict ratings"

    # predict ratings for all

    # Select just test users in training data
    user_vectors = np.asfortranarray([user_vectors[index] for index in user_ids])
    #sparse_train = sparse_train[user_ids, :]
    user_vectors = np.array(user_vectors)

    gc.collect()
    print "generate ratings using gpu"


    #scores = user_vectors.dot(song_vectors.T)

    print "Remove training ratings"

    print "Recommend"
    recs = []
    print "ponto 1: ", str(datetime.now())

    x = T.matrix('x')
    y = theano.shared(np.array(song_vectors, dtype=np.float32))
    sc = T.dot(x, y.T)
    fscore = theano.function([x], sc)

    batch_size = 5000
    full_size = user_vectors.shape[0]
    ids_used = np.array(ids_used)

    if clear:
        rowssparse = train_set.to_records(index=False)
        dbsparse = np.fromiter(rowssparse, dtype=[('user', int), ('music', int), ('count', int)])
        sparse_matrix = sparse.csr_matrix((dbsparse['count'], (dbsparse['user'], dbsparse['music'])))
        sparse_matrix = sparse_matrix.tocsr()

    for j in range((full_size + batch_size - 1) // batch_size):
        init = j * batch_size
        end = init + batch_size

        if init < full_size:
            scores = fscore(np.array(user_vectors[init:end], dtype=np.float32))
            #scores = np.array(user_vectors[init:end], dtype=np.float32).dot(np.array(song_vectors, dtype=np.float32).T)
            scores = np.array(scores)

            # recommend songs for each user
            count = 0
            ids_batch = list(user_ids[init: end])

            for uid in ids_batch:
                if clear:
                    ids_on_train = sparse_matrix[uid].nonzero()[1]
                else:
                    ids_on_train = []

                sc_user = scores[count][:]
                idx_n_z = np.nonzero(sc_user > 0)[0]
                usr_scores_nz = sc_user[idx_n_z]
                idx_sort = usr_scores_nz.argsort()[::-1]

                idx_sort_preds = idx_n_z[idx_sort]
                ids_orig_recs = [id for id in ids_used[idx_sort_preds] if id not in ids_on_train][:NUM_RECOMENDS]
                recs.append(ids_orig_recs)

                if count % 10000 == 0:
                    temp = init + count
                    print " Rec n:", temp, " - hora: ", str(datetime.now())

                count += 1

    met = compute_all_metrics(user_ids, recs, NUM_RECOMENDS, test_set)

    return met


def transform_songs_to_matrix(songs, ids, size, testids, coldids, iscold, iswmf, nfacts=50):

    idxsongs = []
    idsvalid = []

    for inx, val in enumerate(ids):
        if inx < size:
            if not iscold:
                if val not in coldids:
                    if val in testids:
                        idsvalid.append(val)
                        if iswmf:
                            idxsongs.append(val)
                        else:
                            idxsongs.append(inx)
            else:
                if inx < len(songs):
                    idsvalid.append(val)
                    idxsongs.append(inx)

    matrix = np.zeros((len(idxsongs), nfacts))

    i = 0
    for idx in idxsongs:
        matrix[i] = songs[idx]
        i += 1

    return matrix, idsvalid


def generate_metrics_for_model(test_set, U, matrix_songs, train_set, name, ids_used, clear):

    test_all_metrics_convnet = validation(test_set, U, matrix_songs, train_set, ids_used, clear)

    conv_test_str_all_metric = ''
    for k in test_all_metrics_convnet:
        conv_test_str_all_metric += 'CONVNET Metric: ' + str(k) + ' : ' + '{0:.10f}'.format(test_all_metrics_convnet[k])
    print "Result: ", name, " Values: ", conv_test_str_all_metric

    conf = name + " - " + str(datetime.now())

    # Save result on file
    file = open("/home/matheus/Documents/Dissertacao-Dev/modelos/resultscnn2.txt", "a")
    file2 = open("/home/matheus/Dropbox/resultscnn.txt", "a")

    file_line = conf + " Metrics: " + str(conv_test_str_all_metric)
    file.write(file_line + "\n")
    file2.write(file_line + "\n")

    file.close()
    file2.close()


TEST_BIG_DATASET = True
GEN_MODEL_TEST = True
GEN_MODEL_WMF = False
GEN_MODEL_COLD = False


def generate_metrics_convnet(all_predictions, song_ids, new_u, bool_replace_user_facts):

    str_dataset_def = "FULL "
    if not TEST_BIG_DATASET:
        path_wmf = MODEL_PATH_SMALL
        rows = db.select_all_triples_small()
        str_dataset_def = 'SMALL '
    else:
        path_wmf = MODEL_PATH
        rows = db.select_all_triples_full()

    print "Start Testing: ", str_dataset_def, " DATASET"

    # Load factors from wmf model file
    wmf_file = np.load(path_wmf)
    user_facts = wmf_file['U']

    # Get songs predicted
    song_ids_pred = []
    for inx, val in enumerate(song_ids):
        if inx < len(all_predictions):
            song_ids_pred.append(int(val))

    df = pd.DataFrame.from_records(rows, columns=['userid', 'musicid', 'count', 'train', 'test', 'cold'])
    df.columns = ['user', 'music', 'count', 'train', 'test', 'cold']
    df = df[df['music'].isin(song_ids_pred)]

    train_set = df[df['train'] == 1]
    train_set = train_set[list(['user', 'music', 'count'])]
    train_set = train_set.sort(['user'])

    test_set = df[df['test'] == 1]
    test_set = test_set[list(['user', 'music', 'count'])]
    test_set = test_set.sort(['user'])

    test_songs_ids = list(test_set['music'].unique())
    cold_songs_ids = list(df[df['cold'] == 1]['music'].unique())

    if bool_replace_user_facts:
        U = new_u
        mdname = "MODEL2 "
    else:
        U = user_facts
        mdname = "MODEL1 "

    size_test = len(all_predictions)

    if GEN_MODEL_TEST:
        hard_songs.clear()
        print " Start metrics on convnet - test set"
        matrix_songs, ids_used = transform_songs_to_matrix(all_predictions, song_ids, size_test, test_songs_ids,
                                                           cold_songs_ids, False, False)
        print "qtd musicas teste: ", len(ids_used)
        name = "TEST SET - " + mdname + str_dataset_def
        generate_metrics_for_model(test_set, U, matrix_songs, train_set, name, ids_used, True)

        sh = [(k, v) for v, k in sorted([(v, k) for k, v in hard_songs.items()], reverse=True)]

        w = []

        i = 0
        for k in sh:
            w.append(k[0])
            i += 1
            if i == 2000:
                break
        print w


    if GEN_MODEL_WMF:
        print " Start metrics on wmf - test set"
        song_facts = wmf_file['V']
        matrix_songs, ids_used = transform_songs_to_matrix(song_facts, song_ids, size_test
                                                 , test_songs_ids, cold_songs_ids, False, True)
        name = "WMF SET - " + mdname + str_dataset_def
        generate_metrics_for_model(test_set, user_facts, matrix_songs, train_set, name, ids_used, True)

    if GEN_MODEL_COLD:
        hard_songs.clear()
        print " Start metrics on convnet - cold set"
        cold_set = df[df['cold'] == 1]
        cold_set = cold_set[list(['user', 'music', 'count'])]

        cold_set = cold_set.sort(['user'])

        matrix_songs, ids_used = transform_songs_to_matrix(all_predictions, song_ids, size_test
                                                 , test_songs_ids, cold_songs_ids, True, False)

        name = "COLD SET - " + mdname + str_dataset_def
        generate_metrics_for_model(cold_set, U, matrix_songs, None, name, ids_used, False)

        sh = [(k, v) for v, k in sorted([(v, k) for k, v in hard_songs.items()], reverse=True)]

        w = []

        i = 0
        for k in sh:
            w.append(k[0])
            i += 1
            if i == 2000:
                break
        print "ids das musicas dificeis:"
        print w

    print "END TESTS"
