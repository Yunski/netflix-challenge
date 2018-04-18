import argparse
import itertools
import time
import numpy as np
import scipy.sparse

from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from utils import get_netflix_data

def evaluate(P, Q, test_indices, mean=None, acc_threshold=0.3):
    users, items, ratings = test_indices
    acc, loss = 0.0, 0.0
    for n, (u, i, r) in enumerate(itertools.izip(users, items, ratings)):
        prediction = P[u,:].dot(Q[:,i])
        if mean is not None:
            prediction += mean 
        res = r - prediction
        loss = (loss*n + int(res**2)) / (n+1)
        acc = (acc*n + int(np.abs(res) <= acc_threshold)) / (n+1)
    return acc, np.sqrt(loss) 


def train(n_components, demean, n_samples):
    print("Loading data...")
    movie_titles, ratings, rating_indices, n_users, n_items = get_netflix_data(n_samples=n_samples)
    print("number of users with ratings: {}".format(len(np.unique(rating_indices[:,0]))))
    print("number of movies with ratings: {}".format(len(np.unique(rating_indices[:,1]))))
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True)
    kf.get_n_splits(rating_indices)

    if not n_components:
        components = [5, 10, 15, 20, 30, 50]
        components_loss_path = np.zeros((len(components), n_splits)) 
        print("Finding optimal number of components...")
        for n, n_components in enumerate(components):
            print("n_components: {}".format(n_components))
            for k, (train_index, test_index) in enumerate(kf.split(rating_indices)):
                mean = None
                print("Fold {}".format(k))
                test_indices = rating_indices[test_index]
                test_indices = test_indices[:,0], test_indices[:,1], test_indices[:,2]
                if demean:
                    print("De-mean training data...")
                    train_indices = rating_indices[train_index]
                    mean = np.mean(train_indices[:,2])
                    train_indices = train_indices[:,0], train_indices[:,1], train_indices[:,2] - mean 
                    data_train = scipy.sparse.csr_matrix((train_indices[2], (train_indices[0], train_indices[1])), shape=(n_users, n_items))
                else:
                    user_test_indices, item_test_indices = test_indices[0], test_indices[1]
                    data_train = scipy.sparse.lil_matrix(ratings)
                    data_train[user_test_indices, item_test_indices] = 0
                    data_train = scipy.sparse.csr_matrix(ratings)  
                print("Finished de-meaning.")
                start = time.time()
                print("Fitting...")
                svd = TruncatedSVD(n_components=n_components)
                P = svd.fit_transform(data_train)     
                Q = svd.components_
                acc, loss = evaluate(P, Q, test_indices, mean=mean)
                print("Elapsed time: {:.1f}s".format(time.time()-start))
                print("loss: {:.4f} - acc: {:.4f}".format(loss, acc))
                components_loss_path[n, k] = loss
        mean_loss = np.mean(components_loss_path, axis=1)
        best_k = components[np.argmin(mean_loss)]
        best_loss = np.amin(mean_loss)
        print("best k: {}, best loss: {:.4f}".format(best_k, best_loss))
    else:
        print("Performing cross validation...")
        mean_acc = 0.0
        mean_loss = 0.0
        for k, (train_index, test_index) in enumerate(kf.split(rating_indices)):
            mean = None
            print("Fold {}".format(k))
            test_indices = rating_indices[test_index]
            test_indices = test_indices[:,0], test_indices[:,1], test_indices[:,2]
            if demean:
                print("De-mean training data...")
                train_indices = rating_indices[train_index]
                mean = np.mean(train_indices[:,2])
                train_indices = train_indices[:,0], train_indices[:,1], train_indices[:,2] - mean
                data_train = scipy.sparse.csr_matrix((train_indices[2], (train_indices[0], train_indices[1])), shape=(n_users, n_items))
                print("Finished de-meaning.")
            else:
                user_test_indices, item_test_indices = test_indices[0], test_indices[1]
                data_train = scipy.sparse.lil_matrix(ratings)
                data_train[user_test_indices, item_test_indices] = 0
                data_train = scipy.sparse.csr_matrix(ratings)  
            start = time.time()
            print("fitting...")
            svd = TruncatedSVD(n_components=n_components)
            P = svd.fit_transform(data_train)
            Q = svd.components_
            acc, loss = evaluate(P, Q, test_indices, mean=mean)
            print("Elapsed time: {:.4f}".format(time.time()-start))
            print("loss: {:.4f} - acc: {:.4f}".format(loss, acc))
            mean_acc = (mean_acc*k + acc) / (k+1)  
            mean_loss = (mean_loss*k + loss) / (k+1)
        print("mean loss: {:.4f} - mean acc: {:.4f}".format(mean_loss, mean_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Baseline Models")
    parser.add_argument('-n', help='n_components', dest='n_components', type=int, default=None)
    parser.add_argument('-s', help='size of dataset', dest='n_samples', type=int, default=None)
    parser.add_argument('--demean', help='de-mean', dest='demean', action='store_true')
    args = parser.parse_args()
    train(args.n_components, args.demean, args.n_samples)

