import argparse
import os
import numpy as np
import tensorflow as tf

from sklearn.model_selection import KFold

from mf_tf import MF
from utils import get_netflix_data

"""
SGD Biased MF Script
"""

def train(n_features=None):
    logdir = 'logs/mf/tf'
    if not os.path.exists(logdir):
        os.makedirs(logdir)
 
    print("Loading data...")
    movie_titles, ratings, rating_indices, n_users, n_items = get_netflix_data(n_samples=100000)
    print("number of users: {}".format(n_users))
    print("number of movies: {}".format(n_items-1))    
    n_splits = 5
    latent_features = [50, 30, 20, 15]
    loss_path = np.zeros((len(latent_features), n_splits))
    kf = KFold(n_splits=5, shuffle=True)
    kf.get_n_splits(rating_indices)
    mean_loss = 0.0
    if n_features:
        print("Fitting with {} latent features.".format(n_features))
    else:
        print("Finding optimal number of latent features.")
    for k, (train_index, test_index) in enumerate(kf.split(rating_indices)):
        print("Fold {}".format(k))
        train_indices = rating_indices[train_index]
        test_indices = rating_indices[test_index]
        val_indices = train_indices[:len(test_indices)/2]
        train_indices = train_indices[len(test_indices)/2:]  
        train_indices = train_indices[:,0], train_indices[:,1], train_indices[:,2]
        val_indices = val_indices[:,0], val_indices[:,1], val_indices[:,2]
        test_indices = test_indices[:,0], test_indices[:,1], test_indices[:,2]
        data_indices = train_indices, val_indices, test_indices
        batch_size = len(train_indices[0]) / 20
        if n_features:
            mf = MF(n_users, n_items, n_features, np.mean(train_indices[2]))
            with tf.Session(graph=mf.graph) as sess:
                _, test_loss = mf.fit(sess, data_indices, batch_size=batch_size)
                mean_loss = (mean_loss*k + test_loss) / (k+1)
        else:
            for i, n in enumerate(latent_features):
                print("n_features: {}".format(n))
                mf = MF(n_users, n_items, n, np.mean(train_indices[2]))
                with tf.Session(graph=mf.graph) as sess:
                    _, test_loss = mf.fit(sess, data_indices, batch_size=batch_size)    
                    loss_path[i, k] = test_loss
    if n_features:
        print("mean loss: {:.4f}".format(mean_loss))
    else:
        mean_losses = np.mean(loss_path, axis=1)
        best_k = latent_features[np.argmin(mean_losses)]    
        best_loss = np.amin(mean_losses)
        print("best k: {} - loss: {:.4f}".format(best_k, best_loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Biased MF Tensorflow")
    parser.add_argument('-n', help='n features', dest='n_features', type=int, default=0)
    args = parser.parse_args()
    train(args.n_features)
