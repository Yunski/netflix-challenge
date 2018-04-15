import time
import numpy as np
import scipy.sparse

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import NMF
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors

from utils import get_netflix_data, get_adj_lists, get_train_test_indices


def evaluate(P, Q, test_indices, ratings, acc_threshold=0.1):
    users, items = test_indices
    acc, loss = 0.0, 0.0
    for n, (u, i) in enumerate(zip(users, items)):
        prediction = P[u,:].dot(Q[:,i])
        res = ratings[u,i] - prediction
        loss = (loss*n + int(res**2)) / (n+1)
        acc = (acc*n + int(np.abs(res) <= acc_threshold)) / (n+1)
    return acc, np.sqrt(loss)        


def train():
    print("Loading data...")
    movie_titles, ratings, rating_indices, n_users, n_items = get_netflix_data()
    print("number of users: {}".format(n_users))
    print("number of movies: {}".format(n_items))
    print("Building adjacency lists...")
    user_adj_list, _ =  get_adj_lists(rating_indices)
    print("Generating train-test indices...")
    test_indices = get_train_test_indices(user_adj_list)
    user_test_indices, item_test_indices = test_indices
    data_train = scipy.sparse.lil_matrix(ratings)
    data_train[user_test_indices, item_test_indices] = 0
    data_train = scipy.sparse.csr_matrix(data_train)
    
    models = ['tsvd', 'pca', 'nmf']
    components = [5, 10, 15, 20]
    components_loss_per_model = np.zeros((len(models), len(components))) 
    print("Finding optimal number of components...")
    for n, n_components in enumerate(components):
        print("n_components: {}".format(n_components))
        for m, model in enumerate(models):
            start = time.time()
            print("Training {}...".format(model))
            if model == 'tsvd':
                svd = TruncatedSVD(n_components=n_components)
                P = svd.fit_transform(data_train)     
                Q = svd.components_
                print("Evaluating model...")
                print("Elapsed time: {:.1f}".format(time.time()-start))
                if np.sum(svd.explained_variance_ratio_) > 0.9:
                    components_loss_path[m, n] = float('inf') 
                    continue
            elif model == 'pca':
                pca = PCA(n_components=n_components)
                P = pca.fit_transform(data_train)
                Q = pca.components_
                print("Evaluating model...")
                print("Elapsed time: {:.1f}".format(time.time()-start))
                if np.sum(pca.explained_variance_ratio_) > 0.9:
                    components_loss_path[m, n] = float('inf')
                    continue
            else:
                nmf = NMF(n_components=n_components)
                P = nmf.fit_transform(data_train)
                Q = nmf.components_
                print("Evaluating model...")
                print("Elapsed time: {:.1f}".format(time.time()-start))
            
            acc, loss = evaluate(P, Q, test_indices, ratings)
            print("{} - loss: {:.4f} - acc: {:.4f}".format(model, loss, acc))
            components_loss_path[m, n] = loss

    best_n_components = np.argmin(components_loss_per_model, axis=1)
    for model, ind in zip(models, best_n_components):
        print("{}: {}".format(model, ind))
    return 

    print("Performing cross validation...")
    n_splits = 5
    for k in range(n_splits): 
        print("Split {}".format(k))
        print("Generating train-test indices...")
        test_indices = get_test_indices(ratings)
        user_test_indices, item_test_indices = test_indices 

        data_train = scipy.sparse.lil_matrix(ratings)
        data_train[user_test_indices, item_test_indices] = 0
        data_train = scipy.sparse.csr_matrix(data_train)
        for model in models:
            print("Training {}...".format(model))
            if model == 'tsvd':
                svd = TruncatedSVD(n_components=n_components)
                P = svd.fit_transform(data_train)
                print("explained variance ratio: {}".format(np.sum(svd.explained_variance_ratio_)))
                Q = svd.components_
            elif model == 'pca':
                pca = PCA(n_components=n_components)
                P = pca.fit_transform(data_train)
                Q = pca.components_
                print(pca.explained_variance_ratio_) 
            else:
                nmf = NMF(n_components=10)
                P = nmf.fit_transform(data_train)   
                Q = nmf.components_
                print(nmf.reconstruction_err_)
            acc, loss = evaluate(P, Q, test_indices, ratings)
            print("{} - loss: {:.4f} - acc: {:.4f}".format(model, loss, acc))


if __name__ == '__main__':
    train()