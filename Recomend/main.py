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


train_size = int(train_pct * ratings.shape[0])

train = ratings[:train_size]
validation = ratings[train_size:]

# models settings
n_feature = 10
eval_iters = 20



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



import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import IndexLocator, FixedLocator, MultipleLocator


def parse_log():
    lines = []
    pmf_results = []
    bpmf_results = []
    als_results = []

    with open('log.log') as file:
        lines = [line.rstrip() for line in file]
        for l in lines:
            if l[0:19] == 'INFO:recommend.pmf:':
                t = re.findall(r"[-+]?\d*\.\d+|\d+", l)
                t = [int(t[0]), float(t[1])]
                pmf_results.append(t)
            elif l[0:19] == 'INFO:recommend.bpmf':
                t = re.findall(r"[-+]?\d*\.\d+|\d+", l)
                t = [int(t[0])-1, float(t[1])]
                bpmf_results.append(t)
            elif l[0:19] == 'INFO:recommend.als:':
                t = re.findall(r"[-+]?\d*\.\d+|\d+", l)
                t = [int(t[0]), float(t[1])]
                als_results.append(t)
    return pmf_results, bpmf_results, als_results

pmf, bpmf, als = parse_log()



pmf_x = [n[0] for n in pmf]
pmf_y = [n[1] for n in pmf]


bpmf_x = [n[0] for n in bpmf]
bpmf_y = [n[1] for n in bpmf]


als_x = [n[0] for n in als]
als_y = [n[1] for n in als]




fig = plt.figure(figsize=(7, 4))

ax = fig.add_subplot()
plt.xlabel('Epochs')
plt.ylabel('RMSE')

ax.plot(pmf_x, pmf_y, label='PMF', color='green')

ax.plot(bpmf_x, bpmf_y, label='BPMF', color='red')

ax.plot(als_x, als_y, label='ALS', color='blue')



ax.yaxis.set_major_locator(FixedLocator([0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15]))

ax.xaxis.set_major_locator(MultipleLocator(base=2))

plt.legend(('PMF', 'BPMF', 'ALS'))

plt.show()

print('PMF predict: ', pmf.predict(np.array([[0, i] for i in range(10)])))
print('BPMF predict: ', bpmf.predict(np.array([[0, i] for i in range(10)])))
print('ALS predict: ', als.predict(np.array([[0, i] for i in range(10)])))

















