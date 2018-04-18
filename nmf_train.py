import argparse
import itertools
import time
import numpy as np
import scipy.sparse

from sklearn.decomposition import NMF
from sklearn.model_selection import KFold

from utils import get_netflix_data

def evaluate(P, Q, test_indices, acc_threshold=0.3):
    users, items, ratings = test_indices
    acc, loss = 0.0, 0.0
    for n, (u, i, r) in enumerate(itertools.izip(users, items, ratings)):
        prediction = P[u,:].dot(Q[:,i])
        res = r - prediction
        loss = (loss*n + int(res**2)) / (n+1)
        acc = (acc*n + int(np.abs(res) <= acc_threshold)) / (n+1)
    return acc, np.sqrt(loss) 


def train(n_components, alpha, reg, n_samples):
    print("Loading data...")
    movie_titles, ratings, rating_indices, n_users, n_items = get_netflix_data(n_samples=n_samples)
    print("number of users with ratings: {}".format(len(np.unique(rating_indices[:,0]))))
    print("number of movies with ratings: {}".format(len(np.unique(rating_indices[:,1]))))
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True)
    kf.get_n_splits(rating_indices)
    ratings = scipy.sparse.dok_matrix(ratings)
    if not n_components or not alpha or not reg:
        alphas = [0.01, 0.1, 1, 10]
        regs = [0.01, 0.1]
        components = [5, 10, 15, 20, 30]
        print("Finding optimal parameters...")
        best_loss = float('inf')
        best_k, best_alpha, best_reg = 0, 0, 0
        for n_components in components:
            for alpha in alphas:
                for reg in regs:
                    mean_loss = 0.0
                    print("n_components: {}, alpha: {}, reg: {}".format(n_components, alpha, reg))
                    for k, (train_index, test_index) in enumerate(kf.split(rating_indices)):
                        print("Fold {}".format(k))
                        test_indices = rating_indices[test_index]
                        test_indices = test_indices[:,0], test_indices[:,1], test_indices[:,2]
                        user_test_indices, item_test_indices = test_indices[0], test_indices[1]
                        data_train = scipy.sparse.lil_matrix(ratings)
                        data_train[user_test_indices, item_test_indices] = 0
                        data_train = scipy.sparse.csr_matrix(ratings)
                        start = time.time()
                        print("Fitting...")
                        nmf = NMF(n_components=n_components, alpha=alpha, l1_ratio=reg, init='nndsvd', tol=0.001, verbose=1)
                        P = nmf.fit_transform(data_train)
                        Q = nmf.components_
                        acc, loss = evaluate(P, Q, test_indices)
                        print("Elapsed time: {:.1f}s".format(time.time()-start))
                        print("loss: {:.4f} - acc: {:.4f}".format(loss, acc))
                        mean_loss = (mean_loss*k + loss) / (k+1)
                    if mean_loss < best_loss:
                        best_loss = mean_loss
                        best_k = n_components
                        best_alpha = alpha
                        best_reg = reg
        print("best k: {}, best alpha: {}, best reg: {}, best loss: {:.4f}".format(best_k, best_alpha, best_reg, best_loss))
    else:
        print("Performing cross validation...")
        mean_acc = 0.0
        mean_loss = 0.0
        for k, (train_index, test_index) in enumerate(kf.split(rating_indices)):
            print("Fold {}".format(k))
            test_indices = rating_indices[test_index]
            test_indices = test_indices[:,0], test_indices[:,1], test_indices[:,2]
            user_test_indices, item_test_indices = test_indices[0], test_indices[1]
            data_train = scipy.sparse.lil_matrix(ratings)
            data_train[user_test_indices, item_test_indices] = 0
            data_train = scipy.sparse.csr_matrix(data_train)
            start = time.time()
            print("Fitting...")
            nmf = NMF(n_components=n_components, alpha=alpha, l1_ratio=reg, init='nndsvd', tol=0.001, verbose=1)
            P = nmf.fit_transform(data_train)   
            Q = nmf.components_
            acc, loss = evaluate(P, Q, test_indices)
            print("Elapsed time: {:.4f}".format(time.time()-start))
            print("loss: {:.4f} - acc: {:.4f}".format(loss, acc))
            mean_acc = (mean_acc*k + acc) / (k+1)  
            mean_loss = (mean_loss*k + loss) / (k+1)
        print("mean loss: {:.4f} - mean acc: {:.4f}".format(mean_loss, mean_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Baseline Models")
    parser.add_argument('-n', help='n_components', dest='n_components', type=int, default=None)
    parser.add_argument('--alpha', help='alpha', dest='alpha', type=float, default=None)
    parser.add_argument('--reg', help='reg', dest='reg', type=float, default=None)
    parser.add_argument('-s', help='size of dataset', dest='n_samples', type=int, default=None)
    args = parser.parse_args()
    train(args.n_components, args.alpha, args.reg, args.n_samples)

