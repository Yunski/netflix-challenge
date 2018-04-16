import os
import numpy as np
np.random.seed(424)
import pandas as pd
import scipy.sparse

def get_netflix_data(n_samples=None, data_dir='data'):
    movie_titles = pd.read_csv(os.path.join(data_dir, 'movie_titles.txt'), header=None, names=['ID','Year','Name'])
    ratings_csr = scipy.sparse.load_npz(os.path.join(data_dir, 'netflix_full_csr.npz')).T
    n_users, n_items = ratings_csr.shape
    if n_samples is not None:
        random_sample = np.random.choice(n_users, size=n_samples, replace=False)
        ratings_csr = ratings_csr[random_sample,:]
        n_users, n_items = ratings_csr.shape
    rating_indices = scipy.sparse.find(ratings_csr)
    rating_indices = np.column_stack(rating_indices).astype(np.int64)
    return movie_titles, ratings_csr, rating_indices, n_users, n_items
 

def get_adj_lists(rating_indices, ratings=None):
    provided_ratings = False
    if len(rating_indices) == 2:
        users, items = rating_indices
        provided_ratings = True
    else:
        users, items, ratings = rating_indices
    user_adj_list = {}
    item_adj_list = {}
    if provided_ratings:
        for u, i in zip(users, items):
            r = ratings[u,i]
            if u not in user_adj_list:
                user_adj_list[u] = [(i,r)]
            else:
                user_adj_list[u].append((i,r))
            if i not in item_adj_list:
                item_adj_list[i] = [(u,r)]
            else:
                item_adj_list[i].append((u,r))
    else:
        for u, i, r in zip(users, items, ratings):
            if u not in user_adj_list:
                user_adj_list[u] = [(i,r)]
            else:
                user_adj_list[u].append((i,r)) 
            if i not in item_adj_list:
                item_adj_list[i] = [(u,r)]
            else:
                item_adj_list[i].append((u,r))

    return user_adj_list, item_adj_list


if __name__ == '__main__':
    movie_titles, _, rating_indices, n_items, n_users = get_netflix_data(n_samples=1000)
    user_adj_list, item_adj_list = get_adj_lists(rating_indices, verbose=1)
    print(len(user_adj_list), len(item_adj_list))

