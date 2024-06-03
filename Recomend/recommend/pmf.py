import logging
from six.moves import xrange
import time

import numpy as np
from numpy.random import RandomState
from .base import ModelBase
from .exceptions import NotFittedError
from .utils.validation import check_ratings
from .utils.evaluation import RMSE


logger = logging.getLogger(__name__)


class PMF(ModelBase):
    # Probabilistic Matrix Factorization

    def __init__(self, n_user, n_item, n_feature, n_size=1000000, batch_size=1e5, epsilon=50.0,
                 momentum=0.8, seed=None, reg=1e-2, converge=1e-5,
                 max_rating=None, min_rating=None):

        super(PMF, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.n_feature = n_feature
        self.n_size = n_size

        self.random_state = RandomState(seed)

        self.batch_size = batch_size

        self.epsilon = float(epsilon)
        self.momentum = float(momentum)
        self.reg = reg
        self.converge = converge
        self.max_rating = float(max_rating) \
            if max_rating is not None else max_rating
        self.min_rating = float(min_rating) \
            if min_rating is not None else min_rating

        self.mean_rating_ = None
        self.user_features_ = 0.1 * self.random_state.rand(n_user, n_feature)
        self.item_features_ = 0.1 * self.random_state.rand(n_item, n_feature)

    def fit(self, ratings, n_iters=50):

        check_ratings(ratings, self.n_user, self.n_item,
                      self.max_rating, self.min_rating)

        self.mean_rating_ = np.mean(ratings[:, 2])
        last_rmse = None
        batch_num = int(np.ceil(float(ratings.shape[0] / self.batch_size)))
        logger.debug("batch count = %d", batch_num + 1)

        u_feature_mom = np.zeros((self.n_user, self.n_feature), dtype='float64')
        i_feature_mom = np.zeros((self.n_item, self.n_feature), dtype='float64')
        u_feature_gradients = np.zeros((self.n_user, self.n_feature), dtype='float64')
        i_feature_gradients = np.zeros((self.n_item, self.n_feature), dtype='float64')
        start = time.time()
        for iteration in xrange(n_iters):
            logger.debug("iteration %d...", iteration)

            self.random_state.shuffle(ratings)

            for batch in xrange(batch_num):
                start_idx = int(batch * self.batch_size)
                end_idx = int((batch + 1) * self.batch_size)
                data = ratings[start_idx:end_idx]

                u_features = self.user_features_.take(
                    data.take(0, axis=1), axis=0)
                i_features = self.item_features_.take(
                    data.take(1, axis=1), axis=0)
                predictions = np.sum(u_features * i_features, 1)
                errs = predictions - (data.take(2, axis=1) - self.mean_rating_)
                err_mat = np.tile(2 * errs, (self.n_feature, 1)).T
                u_gradients = i_features * err_mat + self.reg * u_features
                i_gradients = u_features * err_mat + self.reg * i_features

                u_feature_gradients.fill(0.0)
                i_feature_gradients.fill(0.0)
                for i in xrange(data.shape[0]):
                    row = data.take(i, axis=0)
                    u_feature_gradients[row[0], :] += u_gradients.take(i, axis=0)
                    i_feature_gradients[row[1], :] += i_gradients.take(i, axis=0)

                u_feature_mom = (self.momentum * u_feature_mom) + \
                    ((self.epsilon / data.shape[0]) * u_feature_gradients)
                i_feature_mom = (self.momentum * i_feature_mom) + \
                    ((self.epsilon / data.shape[0]) * i_feature_gradients)

                self.user_features_ -= u_feature_mom
                self.item_features_ -= i_feature_mom

            train_predictions = self.predict(ratings[:, :2])
            train_rmse = RMSE(train_predictions, ratings[:, 2])
            end = time.time() - start
            logger.info("iter: %d, train RMSE: %.6f, time: %.6f, size: %d", iteration, train_rmse, end, self.n_size)

        return self

    def predict(self, data):

        if not self.mean_rating_:
            raise NotFittedError()

        u_features = self.user_features_.take(data.take(0, axis=1), axis=0)
        i_features = self.item_features_.take(data.take(1, axis=1), axis=0)
        predictions = np.sum(u_features * i_features, 1) + self.mean_rating_

        if self.max_rating:
            predictions[predictions > self.max_rating] = self.max_rating

        if self.min_rating:
            predictions[predictions < self.min_rating] = self.min_rating
        return predictions
