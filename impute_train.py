import numpy as np
import scipy.sparse

from collections import Counter
from sklearn.model_selection import KFold

from utils import get_netflix_data

def mean_impute(train_indices):
    users, items, ratings = train_indices
    mean_rating_by_item = {}
    global_mean = 0.0
    for count, (u, i, r) in enumerate(zip(users, items, ratings)):
        global_mean = (global_mean*count + r) / (count+1)
        if i not in mean_rating_by_item:
            mean_rating_by_item[i] = float(r), 1
        else:
            rating, n = mean_rating_by_item[i]
            mean_rating_by_item[i] = float(rating*n + r) / (n+1), n+1
    mean_rating_by_item = {i: mean_rating_by_item[i][0] for i in mean_rating_by_item}
    return mean_rating_by_item, global_mean


def median_impute(train_indices):
    users, items, ratings = train_indices
    median_rating_by_item = {}
    all_ratings = []
    for count, (u, i, r) in enumerate(zip(users, items, ratings)):
        all_ratings.append(r)
        if i not in median_rating_by_item:
            median_rating_by_item[i] = [r]
        else:
            median_rating_by_item[i].append(r)
    median_rating_by_item = {i: np.median(median_rating_by_item[i]) for i in median_rating_by_item}
    global_median = np.median(all_ratings)
    return median_rating_by_item, global_median


def mode_impute(train_indices):
    users, items, ratings = train_indices
    master = {}
    global_counter = Counter()
    for u, i, r in zip(users, items, ratings):
        global_counter[r] += 1
        if i not in master:
            master[i] = Counter({r: 1})
        else:
            master[i][r] += 1
    mode_rating_by_user = {i: master[i].most_common()[0][0] for i in master}
    global_mode = global_counter.most_common()[0][0]
    return mode_rating_by_user, global_mode


def evaluate(test_indices, rating_by_item, global_val, acc_threshold=0.3):
    users, items, ratings = test_indices
    acc, loss = 0.0, 0.0
    for n, (u, i, r) in enumerate(zip(users, items, ratings)):
        if i in rating_by_item:
            prediction = rating_by_item[i]
        else:
            prediction = global_val
        res = r - prediction
        loss = (loss*n + res**2) / (n+1)
        acc = (acc*n + int(np.abs(res) <= acc_threshold)) / (n+1)
    return acc, np.sqrt(loss)


def train():
    print("Loading data...")
    movie_titles, _, rating_indices, n_users, n_items = get_netflix_data(n_samples=1000000)
    print("number of users: {}".format(n_users))
    print("number of movies: {}".format(n_items-1))

    print("Performing cross validation...")
    strategies = ['mean', 'mode', 'median']
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True)
    kf.get_n_splits(rating_indices)

    loss_path = np.zeros((len(strategies), n_splits))
    acc_path = np.zeros((len(strategies), n_splits))
    for k, (train_index, test_index) in enumerate(kf.split(rating_indices)):
        print("Fold {}".format(k))
        train_indices, test_indices = rating_indices[train_index], rating_indices[test_index]
        train_indices = (train_indices[:,0], train_indices[:,1], train_indices[:,2])
        test_indices = (test_indices[:,0], test_indices[:,1], test_indices[:,2])
        for i, strategy in enumerate(strategies):
            print("strategy: {}".format(strategy))
            if strategy == 'mean':
                rating_by_item, global_val = mean_impute(train_indices)
            elif strategy == 'mode':
                rating_by_item, global_val = mode_impute(train_indices)
            else:
                rating_by_item, global_val = median_impute(train_indices)
            acc, loss = evaluate(test_indices, rating_by_item, global_val)
            print("{} - loss: {:.4f} - acc: {:.4f}".format(strategy, loss, acc))
            acc_path[i, k] = acc
            loss_path[i, k] = loss

    mean_acc_by_strategy = np.mean(acc_path, axis=1)
    print("mean acc: {:.4f}(mean), {:.4f}(mode), {:.4f}(median)".format(mean_acc_by_strategy[0],
                                                            mean_acc_by_strategy[1],
                                                            mean_acc_by_strategy[2]))
    mean_loss_by_strategy = np.mean(loss_path,axis=1)
    print("mean loss: {:.4f}(mean), {:.4f}(mode) {:.4f}(median)".format(mean_loss_by_strategy[0], 
                                                            mean_loss_by_strategy[1],
                                                            mean_loss_by_strategy[2]))
    if np.argmin(mean_loss_by_strategy) == 0:
        print("best strategy is mean")
    elif np.argmin(mean_loss_by_strategy) == 1:
        print("best strategy is mode")
    else:
        print("best strategy is median")

    
if __name__ == '__main__':
    train()    
