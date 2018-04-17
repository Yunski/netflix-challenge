import numpy as np
import pandas as pd
import scipy.sparse

from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

# This file consists of titles and release years associated with each ID
movie_titles = pd.read_csv('data/movie_titles.txt', header = None, names = ['ID','Year','Name'])
print(movie_titles.head())

# This file is a sparse matrix of movies by user, with each element a rating (1-5) or nonresponse (0)
ratings_csr = scipy.sparse.load_npz('data/netflix_full_csr.npz').T
rating_indices = scipy.sparse.find(ratings_csr)
print(ratings_csr.shape)
print("users with ratings: {}, movies with ratings: {}".format(len(np.unique(rating_indices[0])), 
                                                               len(np.unique(rating_indices[1]))))
print("Fitting tsvd...")
n_components = 15
svd = TruncatedSVD(n_components = n_components)
P = svd.fit_transform(ratings_csr)
Q = svd.components_
print(svd.explained_variance_ratio_)

print("Evaluating tsvd...")
user_indices, item_indices, ratings = rating_indices
loss = 0

for n, (u, i, r) in enumerate(zip(user_indices, item_indices, ratings)):
    res = r - P[u,:].dot(Q[:,i])
    loss = (loss*n + res**2) / (n+1)

print("loss: {:.4f}".format(np.sqrt(loss)))

