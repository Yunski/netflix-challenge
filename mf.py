import os
import numpy as np

from tqdm import tqdm

from utils import get_adj_lists

"""
Implementation of "Matrix Factorization Techniques for Recommender Systems"
with scikit-like api

References https://lazyprogrammer.me/tutorial-on-collaborative-filtering-and-matrix-factorization-in-python/
and https://blog.insightdatascience.com/explicit-matrix-factorization-als-sgd-and-all-that-jazz-b00e4d9b21ea
"""

class MF(object):
    def __init__(self, n_users, n_items, n_features, method='sgd', learning_rate=0.01, reg=0.1, acc_threshold=0.3):
        if method not in ['als', 'sgd']:
            raise ValueError("method not supported: {}".format(method))
        self.name = "mf"
        self.n_users = n_users
        self.n_items = n_items
        self.n_features = n_features
        self.method = method
        self.learning_rate = learning_rate
        self.reg = reg
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


    def _als(self, user_adj_list, item_adj_list):
        self._update_user_params(user_adj_list)            
        self._update_item_params(item_adj_list)            

        
    def _sgd(self, train_indices):
        user_indices, item_indices, ratings = train_indices
        for u, i, r in zip(user_indices, item_indices, ratings):
            res = r - self._predict(u, i)
            self.b_u[u] += self.learning_rate * (res - self.reg * self.b_u[u])
            self.b_i[i] += self.learning_rate * (res - self.reg * self.b_i[i]) 
            self.P[u,:] += self.learning_rate * (res * self.Q[:,i] - self.reg * self.P[u,:])
            self.Q[:,i] += self.learning_rate * (res * self.P[u,:] - self.reg * self.Q[:,i])


    def fit(self, train_indices, max_iter=100, tol=1e-4, save_model=True, verbose=0):
        if self.method == 'als':
            if verbose:
                print("Building adjacency list...")
            user_adj_list, item_adj_list = get_adj_lists(train_indices)
        self.mu = np.mean(train_indices[2])
        cur_loss = 0
        prev_loss = 0
        
        if verbose:
            print("Fitting with {}...".format(self.method))
        for i in range(max_iter): 
            print("iter {}".format(i))
            if self.method == 'als':
                self._als(user_adj_list, item_adj_list)
            else:
                self._sgd(train_indices)

            acc, loss = self.eval(train_indices)
            print("train_loss: {:.4f} - train_err: {:.4f} - train_acc: {:.4f}".format(loss, 1-acc, acc))
            prev_loss = cur_loss
            cur_loss = loss
            if np.abs(cur_loss - prev_loss) <= tol:
                break
           
        if save_model:
            save_path = 'logs/mf/numpy'
            np.save(os.path.join(save_path, 'P'), self.P)
            np.save(os.path.join(save_path, 'Q'), self.Q)
            np.save(os.path.join(save_path, 'b_u'), self.b_u)
            np.save(os.path.join(save_path, 'b_i'), self.b_i)

        if verbose:
            print("Training finished.")    
        
        return cur_loss


    def predict(self, test_indices):
        return self.eval(test_indices)


    def _predict(self, u, i):
        return self.P[u,:].dot(self.Q[:,i]) + self.b_u[u] + self.b_i[i] + self.mu


    def eval(self, indices):
        user_indices, item_indices, ratings = indices
        acc, loss = 0.0, 0.0
        for n, (u, i, r) in enumerate(zip(user_indices, item_indices, ratings)):
            prediction = self._predict(u, i)
            res = r - prediction 
            loss = (loss*n + res**2) / (n+1)
            acc = (acc*n + int(np.abs(res) <= self.acc_threshold)) / (n+1) 

        return acc, np.sqrt(loss)
