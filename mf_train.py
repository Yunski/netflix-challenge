import argparse
import os
import time
import numpy as np
import scipy.sparse

from sklearn.model_selection import KFold

from mf import MF
from utils import get_netflix_data

"""
Matrix Factorization Train Script
"""

def train(reg):
    logdir = 'logs/mf/numpy'
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    print("Loading data...")
    movie_titles, ratings, rating_indices, n_users, n_items = get_netflix_data(n_samples=1000000)
    print("number of users with ratings: {}".format(len(np.unique(rating_indices[:,0]))))
    print("number of movies with ratings: {}".format(len(np.unique(rating_indices[:,1]))))

    method = 'als'
    if reg:
        print("Performing cross validation with reg: {}.".format(reg))
    else:
        print("Finding optimal regularization penalty.")
    reg_vals = [0.01, 0.1, 1, 10]
    best_reg = 0
    mean_loss = 0.0
    n_splits = 5
    n_features = 15
    loss_path = np.zeros((len(reg_vals), n_splits))
    kf = KFold(n_splits=n_splits, shuffle=True)
    kf.get_n_splits(rating_indices)
    for k, (train_index, test_index) in enumerate(kf.split(rating_indices)):
        print("Fold {}".format(k))
        train_indices, test_indices = rating_indices[train_index], rating_indices[test_index]
        train_indices = (train_indices[:,0], train_indices[:,1], train_indices[:,2])
        test_indices = (test_indices[:,0], test_indices[:,1], test_indices[:,2])
        if reg:
            start = time.time() 
            model = MF(n_users, n_items, n_features, method=method)
            model.fit(train_indices, verbose=1) 
            acc, loss = model.predict(test_indices)
            print("val_loss: {:.4f} - val_acc: {:.4f}".format(loss, acc))
            mean_loss = (mean_loss*k + loss) / (k+1)
        else:
            for i, reg in enumerate(reg_vals):
                print("lambda: {}".format(reg))
                start = time.time()
                model = MF(n_users, n_items, n_features, method=method)
                model.fit(train_indices, verbose=1)
                acc, loss = model.predict(test_indices)
                print("val_loss: {:.4f} - val_acc: {:.4f}".format(loss, acc))
                loss_path[i, k] = loss
    if reg:
        print("mean loss: {:.4f}".format(mean_loss))
    else:
        loss_means = np.mean(loss_path, axis=1)
        print(loss_means)
        best_reg = reg_vals[np.argmin(loss_means)]
        best_loss = np.amin(loss_means)
        print("best lambda: {} - loss: {}".format(best_reg, best_loss))
    print("Successfully finished training MF. See logs directory.")
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Biased MF Numpy")
    parser.add_argument('-n', help='n features', dest='n_features', type=int, default=0) 
    args = parser.parse_args()
    train(args.n_features)
   
