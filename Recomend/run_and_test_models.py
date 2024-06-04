import numpy as np
from recommend.bpmf import BPMF
from recommend.pmf import PMF
from recommend.als import ALS

from recommend.utils.evaluation import RMSE
from recommend.utils.datasets import load_movielens_1m_ratings
from numpy.random import RandomState

import logging

logging.basicConfig(filename='log.log', level=logging.INFO, filemode='w')
logger = logging.getLogger(__name__)

# load user ratings
ratings = load_movielens_1m_ratings('ml-1m/ratings.dat')

n_user = max(ratings[:, 0])
n_item = max(ratings[:, 1])
ratings[:, (0, 1)] -= 1 # shift ids by 1 to let user_id & movie_id start from 0

rand_state = RandomState(0)

# split data to training & testing
train_pct = 0.9
rand_state.shuffle(ratings)

#ratings = ratings[0:int(len(ratings)*0.001)]

train_size = int(train_pct * ratings.shape[0])

train = ratings[:train_size]
validation = ratings[train_size:]

# models settings
n_feature = 10
eval_iters = 30


print("pmf  n_user: %d, n_item: %d, n_feature: %d, training size: %d, validation size: %d" % (
    n_user, n_item, n_feature, train.shape[0], validation.shape[0]))
pmf = PMF(n_user=n_user, n_item=n_item, n_feature=n_feature,
          epsilon=25., max_rating=5., min_rating=1., seed=0)

pmf.fit(train, n_iters=eval_iters)
train_preds = pmf.predict(train[:, :2])
train_rmse = RMSE(train_preds, train[:, 2])
val_preds = pmf.predict(validation[:, :2])
val_rmse = RMSE(val_preds, validation[:, 2])
print("after %d iterations, train RMSE: %.6f, validation RMSE: %.6f" % \
      (eval_iters, train_rmse, val_rmse))

print('\n\n')



print("bpmf n_user: %d, n_item: %d, n_feature: %d, training size: %d, validation size: %d" % (
    n_user, n_item, n_feature, train.shape[0], validation.shape[0]))
bpmf = BPMF(n_user=n_user, n_item=n_item, n_feature=n_feature,
            max_rating=5., min_rating=1., seed=0)

bpmf.fit(train, n_iters=eval_iters)
train_preds = bpmf.predict(train[:, :2])
train_rmse = RMSE(train_preds, train[:, 2])
val_preds = bpmf.predict(validation[:, :2])
val_rmse = RMSE(val_preds, validation[:, 2])
print("after %d iteration, train RMSE: %.6f, validation RMSE: %.6f" %
      (eval_iters, train_rmse, val_rmse))


print('\n\n')



print("als  n_user: %d, n_item: %d, n_feature: %d, training size: %d, validation size: %d" % (
    n_user, n_item, n_feature, train.shape[0], validation.shape[0]))
als = ALS(n_user=n_user, n_item=n_item, n_feature=n_feature,
          reg=5e-2, max_rating=5., min_rating=1., seed=0)

als.fit(train, n_iters=eval_iters)
train_preds = als.predict(train[:, :2])
train_rmse = RMSE(train_preds, train[:, 2])
val_preds = als.predict(validation[:, :2])
val_rmse = RMSE(val_preds, validation[:, 2])
print("after %d iterations, train RMSE: %.6f, validation RMSE: %.6f" % \
      (eval_iters, train_rmse, val_rmse))


user = int(input('Input index of user: '))
n_films = int(input('Input number of films to recommend: '))

arr = als.predict(np.array([[user, i] for i in range(n_item)]))
arr = [[i, k] for i, k in zip(arr, range(n_item))]
arr = sorted(arr,key=lambda x: x[0], reverse=True)

films = ratings.take([1], axis=1)



for i in arr:
    if i[1] in films:
        arr.remove(i)


arr = [i[1] for i in arr]

print('Indexes of recommended films:\n ', arr[0:n_films])
#print(arr)














