import os
import numpy as np

from tqdm import tqdm

from config import cfg
from utils import get_adj_lists

"""
Implementation of "Matrix Factorization Techniques for Recommender Systems"
with scikit-like api

References https://lazyprogrammer.me/tutorial-on-collaborative-filtering-and-matrix-factorization-in-python/
"""

class MF(object):
    def __init__(self, n_users, n_items, n_features, mu, reg=0.1, acc_threshold=0.1):
        self.name = "mf"
        self.n_users = n_users
        self.n_items = n_items
        self.n_features = n_features
        self.reg = reg
        self.mu = mu
        self.acc_threshold = acc_threshold
        
        self._init_all_variables()


    def _init_all_variables(self):
        self.b_u = np.zeros(self.n_users)
        self.b_i = np.zeros(self.n_items)
        self.P = np.random.randn(self.n_users, self.n_features) / self.n_features
        self.Q = np.random.randn(self.n_features, self.n_items) / self.n_features
    

    def _update_user_params(self, user_adj_list):
        for u, item_ratings in user_adj_list.iteritems():
            res_loss = 0
            for i, r in item_ratings:
                res_loss += r - self.P[u,:].dot(self.Q[:,i]) - self.b_i[i] - self.mu 
            self.b_u[u] = res_loss / (1 + self.reg) / len(user_adj_list[u])
        for u, item_ratings in user_adj_list.iteritems(): 
            X = self.reg * np.eye(self.n_features)
            v = np.zeros(self.n_features)
            for i, r in item_ratings:
                X += np.outer(self.Q[:,i], self.Q[:,i])
                v += (r - self.b_u[u] - self.b_i[i] - self.mu)*self.Q[:,i]
            self.P[u,:] = np.linalg.solve(X, v) # np.linalg.solve(A, b) returns x for Ax = b                


    def _update_item_params(self, item_adj_list):
        for i, user_ratings in item_adj_list.iteritems():
            res_loss = 0
            for u, r in user_ratings:
                res_loss += r - self.P[u,:].dot(self.Q[:,i]) - self.b_u[u] - self.mu
            self.b_i[i] = res_loss / (1 + self.reg) / len(item_adj_list[i])
        for i, user_ratings in item_adj_list.iteritems():
            X = self.reg * np.eye(self.n_features)
            v = np.zeros(self.n_features)
            for u, r in user_ratings:
                X += np.outer(self.P[u,:], self.P[u,:])        
                v += (r - self.b_u[u] - self.b_i[i] - self.mu)*self.P[u,:]    
            self.Q[:,i] = np.linalg.solve(X, v)


    def fit(self, rating_indices, max_iter=20, tol=1e-2, save_model=True, verbose=0):
        if verbose:
            print("Building adjacency list...")
        user_adj_list, item_adj_list = get_adj_lists(rating_indices)

        cur_loss = 0
        prev_loss = 0
        
        if verbose:
            print("Fitting with alternating least squares...")
        for i in range(max_iter): 
            print("iter {}".format(i))
            self._update_user_params(user_adj_list)            
            self._update_item_params(item_adj_list)            
            acc, loss = self.eval(user_adj_list)
            print("train_loss: {:.4f} - train_err: {:.4f} - train_acc: {:.4f}".format(loss, 1-acc, acc))
            prev_loss = cur_loss
            cur_loss = loss
            if np.abs(loss - prev_loss) <= tol:
                break
           
        if save_model:
            save_path = os.path.join(cfg.logdir, self.name)
            np.save(os.path.join(save_path, 'P'), self.P)
            np.save(os.path.join(save_path, 'Q'), self.Q)
            np.save(os.path.join(save_path, 'b_u'), self.b_u)
            np.save(os.path.join(save_path, 'b_i'), self.b_i)

        if verbose:
            print("Training finished.")    
        
        return cur_loss


    def predict(self, rating_indices):
        user_adj_list, _ = get_adj_lists(rating_indices)
        return self.eval(user_adj_list)


    def _predict(self, u, i):
        return self.P[u,:].dot(self.Q[:,i]) + self.b_u[u] + self.b_i[i] + self.mu


    def eval(self, user_adj_list):
        acc, loss = 0.0, 0.0
        n = 0
        for u, item_ratings in user_adj_list.iteritems():
            for i, r in item_ratings:
                prediction = self._predict(u, i)
                loss += (r-prediction)**2
                ind = int(np.abs(r-prediction) <= self.acc_threshold)
                acc = (acc*n + ind) / (n+1)
                n += 1
        return acc, loss
