import os
import numpy as np
np.random.seed(424)
import pandas as pd
import scipy.sparse

def get_netflix_data(n_samples=1000000, data_dir='data'):
    movie_titles = pd.read_csv(os.path.join(data_dir, 'movie_titles.txt'), header=None, names=['ID','Year','Name'])
    ratings_csr = scipy.sparse.load_npz(os.path.join(data_dir, 'netflix_full_csr.npz')).T
    n_users, n_items = ratings_csr.shape
    if n_samples != None:
        random_sample = np.random.choice(n_users, size=n_samples, replace=False)
        ratings_csr = ratings_csr[random_sample,:]
        n_users, n_items = ratings_csr.shape
    rating_indices = scipy.sparse.find(ratings_csr)      
    rating_indices = np.column_stack(rating_indices).astype(np.int64) 
    return movie_titles, ratings_csr, rating_indices, n_users, n_items-1
 

def get_test_indices(ratings, test_size=0.1):
    test_indices = ([], [])
    n_user, n_items = ratings.shape
    for u in range(n_user):
        rated_items = ratings[u, :].nonzero()[1]
        if len(rated_items) <= 1:
            continue
        size = max(1, int(test_size*len(rated_items)))
        item_indices = np.random.choice(rated_items, size=size, replace=False)
        for i in item_indices:
            test_indices[0].append(u)
            test_indices[1].append(i)
    return test_indices

def get_train_test_indices(user_adj_list, test_size=0.2):
    #train_indices = ([],[],[])
    test_indices = ([],[])
    for u, item_ratings in user_adj_list.iteritems():
        if len(item_ratings) <= 1:
            continue
        size = np.maximum(1, int(test_size*len(item_ratings)))
        item_ratings_copy = list(item_ratings)
        np.random.shuffle(item_ratings_copy)
        for item_rating in item_ratings_copy[:size]:
            test_indices[0].append(u)
            test_indices[1].append(item_rating[0])
            #test_indices[2].append(item_rating[1])
        #for item_rating in item_ratings_copy[size:]:
        #    train_indices[0].append(u)
        #    train_indices[1].append(item_rating[0])
        #    train_indices[2].append(item_rating[1])
    return test_indices


def get_adj_lists(rating_indices):
    users = rating_indices[:,0]
    items = rating_indices[:,1]-1
    ratings = rating_indices[:,2]
    user_adj_list = {}
    item_adj_list = {}
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


def get_batch(batch_size, user_indices, item_indices, ratings):
    indices = np.random.permutation(len(user_indices))[:batch_size]
    return user_indices[indices], item_indices[indices], ratings[indices]


if __name__ == '__main__':
    movie_titles, _, rating_indices, n_items, n_users = get_netflix_data()
    user_adj_list, item_adj_list = get_adj_lists(rating_indices, verbose=1)
    print(len(user_adj_list), len(item_adj_list))

