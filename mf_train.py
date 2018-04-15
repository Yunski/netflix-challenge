import os
import time
import numpy as np

from sklearn.model_selection import KFold

from config import cfg
from mf import MF
from utils import get_netflix_data, get_adj_lists

"""
Matrix Factorization Train Script
"""

def train():
    np.random.seed(424)
    logdir = os.path.join(cfg.logdir, 'mf')
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    print("Loading data...")
    movie_titles, _, ratings, n_users, n_items = get_netflix_data()
    print("number of unique users: {}".format(len(np.unique(ratings[:,0]))))
    print("number of unique movies: {}".format(len(np.unique(ratings[:,1]))))

    print("Performing cross validation...")
    kf = KFold(n_splits=cfg.k_folds, shuffle=True)
    kf.get_n_splits(rating_indices)
    reg_vals = [0.01, 0.1, 1, 10]

    best_reg = 0
    best_loss = float('inf')

    for reg in reg_vals:
        mean_loss = 0
        print("lambda: {}".format(reg))
        for k, (train_index, test_index) in enumerate(kf.split(ratings)):
            print("Fold {}".format(k))
            start = time.time()
            data_train = ratings[train_index]
            data_test = ratings[test_index]
            
            model = MF(n_users, n_items, 30, np.mean(data_train[:,2]))
            model.fit(data_train, verbose=1)

            acc, loss = model.predict(data_test)
            print("elapsed time for fold: {}".format(time.time()-start))
            print("val_loss: {:.4f} - val_acc: {:.4f}".format(loss, acc))
            mean_loss = (mean_loss*k + loss) / (k+1)

        if mean_loss < best_loss:
            best_loss = mean_loss
            best_reg = reg
        else:
            break

    print("best lambda: {} - loss: {}".format(best_reg, best_loss))
    print("Successfully finished training MF. See logs directory.")
 

if __name__ == '__main__':
    train()
   
