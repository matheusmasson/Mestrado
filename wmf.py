import numpy as np
from scipy.sparse import csr_matrix


def update(indices, H, HH, alpha=1.0, lbda=0.015, n_facts=50):
    """
    Update latent factors for a single user or item.
    """
    Hix = H[indices, :]
    M = HH + alpha * Hix.T.dot(Hix) + np.diag(lbda*np.ones(n_facts))
    return np.dot(np.linalg.inv(M), (1+alpha) * Hix.sum(axis=0))


def factorize_only_users(V, iter, sparse, n_facts=50):

    sparse = sparse.tocsr()

    num_users, num_items = sparse.shape
    U = np.empty((num_users, n_facts))

    for it in xrange(iter):
        # fit user factors
        VV = V.T.dot(V)
        for u in range(num_users):
            # get (positive i.e. non-zero scored) items for user
            indices = sparse[u].nonzero()[1]
            if indices.size:
                U[u, :] = update(indices, V, VV)
            else:
                U[u, :] = np.zeros(n_facts)

    return U


def runwmf(U, V, iter, sparse, n_facts=50):

    sparse = sparse.tocsr()

    temp_idc_m = sparse.copy()
    nn_elems = int(sparse.indptr[-1])
    temp_idc_m.data = np.arange(nn_elems)
    col_view_matrix = temp_idc_m.tocsc()

    num_users, num_items = sparse.shape

    for it in xrange(iter):
        print 'wmf iteration', it
        # fit user factors
        VV = V.T.dot(V)
        for u in xrange(num_users):
            # get (positive i.e. non-zero scored) items for user
            indices = sparse[u].nonzero()[1]
            if indices.size:
                U[u,:] = update(indices,V,VV)
            else:
                U[u,:] = np.zeros(n_facts)
        # fit item factors
        UU = U.T.dot(U)
        for i in xrange(num_items):
            col = col_view_matrix[:,i].copy()
            col.data = sparse.data[col.data]

            indices = col.nonzero()[0]
            if indices.size:
                V[i,:] = update(indices, U, UU)
            else:
                V[i,:] = np.zeros(n_facts)
    return U, V


def factorize_only_songs(U, iter, sparse, n_facts=50):

    sparse = sparse.tocsr()

    temp_idc_m = sparse.copy()
    nn_elems = int(sparse.indptr[-1])
    temp_idc_m.data = np.arange(nn_elems)
    col_view_matrix = temp_idc_m.tocsc()

    num_users, num_items = sparse.shape

    V = n_facts**-0.5*np.random.random_sample((num_items, n_facts))

    for it in xrange(iter):
        print 'wmf iteration', it
        UU = U.T.dot(U)
        for i in xrange(num_items):
            col = col_view_matrix[:,i].copy()
            col.data = sparse.data[col.data]

            indices = col.nonzero()[0]
            if indices.size:
                V[i,:] = update(indices, U, UU)
            else:
                V[i,:] = np.zeros(n_facts)
    return V
