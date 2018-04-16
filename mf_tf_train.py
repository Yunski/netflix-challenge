import os
import numpy as np

from sklearn.model_selection import KFold

from mf_tf import MF
from utils import get_netflix_data

"""
SGD Biased MF Script
"""

def train():
    logdir = 'logs/mf/tf'
    if not os.path.exists(logdir):
        os.makedirs(logdir)
 
    print("Loading data...")
    movie_titles, ratings, rating_indices, n_users, n_items = get_netflix_data(n_samples=1000000)
    print("number of users: {}".format(n_users))
    print("number of movies: {}".format(n_items-1))    
    n_splits = 5
    latent_features = [5, 10]
    loss_path = np.zeros((len(latent_features), n_splits))
    kf = KFold(n_splits=10)
    kf.get_n_splits(rating_indices)
    for k, (train_index, test_index) in enumerate(kf.split(rating_indices)):
        print("Fold {}".format(k))
        train_indices = rating_indices[train_index]
        test_indices = rating_indices[test_index]
        train_indices = train_indices[:,0], train_indices[:,1], train_indices[:,2]
        test_indices = test_indices[:,0], test_indices[:,1], test_indices[:,2]
        for i, n_features in enumerate(latent_features):
            print("n_features: {}".format(n_features))
            mf = MF(n_users, n_items, n_features, np.mean(train_indices[2]))
            loss = mf.fit(train_indices, test_indices)    
            loss_path[i, k] = loss

    mean_losses = np.mean(loss_path, axis=1)
    best_k = latent_features[np.argmin(mean_losses)]    
    best_loss = np.amin(mean_losses)
    print("best k: {} - loss: {:.4f}".format(best_k, best_loss))

if __name__ == '__main__':
    train()