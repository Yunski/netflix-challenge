import argparse
import time
import numpy as np
import scipy.sparse

from sklearn.decomposition import NMF
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from utils import get_netflix_data

def evaluate(P, Q, test_indices, ratings, acc_threshold=0.3):
    users, items = test_indices[:,0], test_indices[:,1]
    acc, loss = 0.0, 0.0
    predictions = np.array([P[u,:].dot(Q[:,i]) for u, i in zip(users, items)])
    for n, (u, i) in enumerate(zip(users, items)):
        prediction = P[u,:].dot(Q[:,i])
        res = ratings[u,i] - prediction
        loss = (loss*n + int(res**2)) / (n+1)
        acc = (acc*n + int(np.abs(res) <= acc_threshold)) / (n+1)
    return acc, np.sqrt(loss) 


def train(n_components):
    print("Loading data...")
    movie_titles, ratings, rating_indices, n_users, n_items = get_netflix_data(n_samples=10000)
    print("number of users: {}".format(n_users))
    print("number of movies: {}".format(n_items-1))
    
    models = ['tsvd', 'nmf']
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True)
    kf.get_n_splits(rating_indices)
    ratings = scipy.sparse.dok_matrix(ratings)

    if not n_components:
        components = [5, 10, 15, 20, 30, 50]
        components_loss_per_model = np.zeros((len(models), len(components))) 
        print("Finding optimal number of components...")
        for m, model in enumerate(models):
            print("Training {}...".format(model))
            for n, n_components in enumerate(components):
                print("n_components: {}".format(n_components))
                mean_loss = 0.0 
                for k, (train_index, test_index) in enumerate(kf.split(rating_indices)):
                    print("Fold {}".format(k))
                    test_indices = rating_indices[test_index]
                    user_test_indices, item_test_indices = test_indices[:,0], test_indices[:,1]
                    data_train = scipy.sparse.lil_matrix(ratings)
                    data_train[user_test_indices, item_test_indices] = 0
                    data_train = scipy.sparse.csr_matrix(data_train)
                    start = time.time()
                    if model == 'tsvd':
                        svd = TruncatedSVD(n_components=n_components)
                        P = svd.fit_transform(data_train)     
                        Q = svd.components_
                    else:
                        nmf = NMF(n_components=n_components, init='nndsvd')
                        P = nmf.fit_transform(data_train)
                        Q = nmf.components_
                    
                    acc, loss = evaluate(P, Q, test_indices, ratings)
                    mean_loss = (mean_loss*k + loss) / (k+1)
                    print("Elapsed time: {:.1f}s".format(time.time()-start))
                    print("{} - loss: {:.4f} - acc: {:.4f}".format(model, loss, acc))
                components_loss_per_model[m, n] = mean_loss

        best_n_components = np.argmin(components_loss_per_model, axis=1)
        for model, ind in zip(models, best_n_components):
            print("{}: {}".format(model, components[ind]))
    else:
        print("Performing cross validation...")
        mean_loss = [0.0, 0.0]
        for k, (train_index, test_index) in enumerate(kf.split(rating_indices)):
            print("Fold {}".format(k))
            test_indices = rating_indices[test_index]
            user_test_indices, item_test_indices = test_indices[:,0], test_indices[:,1]
            data_train = scipy.sparse.lil_matrix(ratings)
            data_train[user_test_indices, item_test_indices] = 0
            data_train = scipy.sparse.csr_matrix(data_train)
            for i, model in enumerate(models):
                start = time.time()
                print("Training {}...".format(model))
                if model == 'tsvd':
                    svd = TruncatedSVD(n_components=n_components)
                    P = svd.fit_transform(data_train)
                    Q = svd.components_
                else:
                    nmf = NMF(n_components=n_components, init='nndsvd')
                    P = nmf.fit_transform(data_train)   
                    Q = nmf.components_
                acc, loss = evaluate(P, Q, test_indices, ratings)
                print("Elapsed time: {:.4f}".format(time.time()-start))
                print("{} - loss: {:.4f} - acc: {:.4f}".format(model, loss, acc))
                mean_loss[i] = (mean_loss[i]*k + loss) / (k+1)

        print("mean loss: tsvd {:.4f} - nmf {:.4f}".format(mean_loss[0], mean_loss[1]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Baseline Models")
    parser.add_argument('-k', help='n_components', dest='n_components', type=int, default=None)
    args = parser.parse_args()
    train(args.n_components)
