import logging
from six.moves import xrange
import numpy as np
from numpy.random import RandomState
from numpy.linalg import inv

import time

from .base import ModelBase
from .exceptions import NotFittedError
from .utils.datasets import build_user_item_matrix
from .utils.validation import check_ratings
from .utils.evaluation import RMSE

logger = logging.getLogger(__name__)


class ALS(ModelBase):
    #Alternating Least Squares with Weighted Lambda Regularization (ALS-WR)

    def __init__(self, n_user, n_item, n_feature, n_size=1000000, reg=1e-2, converge=1e-5,
                 seed=None, max_rating=None, min_rating=None):
        super(ALS, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.n_feature = n_feature
        self.n_size = n_size
        self.reg = float(reg)
        self.rand_state = RandomState(seed)
        self.max_rating = float(max_rating) if max_rating is not None else None
        self.min_rating = float(min_rating) if min_rating is not None else None
        self.converge = converge

        self.mean_rating_ = None
        self.ratings_csr_ = None
        self.ratings_csc_ = None

        self.user_features_ = 0.1 * self.rand_state.rand(n_user, n_feature)
        self.item_features_ = 0.1 * self.rand_state.rand(n_item, n_feature)

    def _update_item_feature(self):

        for j in xrange(self.n_item):
            user_idx, _ = self.ratings_csc_[:, j].nonzero()
            n_i = user_idx.shape[0]
            if n_i == 0:
                logger.debug("no ratings for item %d", j)
                continue
            user_features = self.user_features_.take(user_idx, axis=0)
            ratings = self.ratings_csc_[:, j].data - self.mean_rating_

            A_j = (np.dot(user_features.T, user_features) +
                   self.reg * n_i * np.eye(self.n_feature))
            V_j = np.dot(user_features.T, ratings)
            self.item_features_[j, :] = np.dot(inv(A_j), V_j)

    def _update_user_feature(self):
        for i in xrange(self.n_user):
            _, item_idx = self.ratings_csr_[i, :].nonzero()
            n_u = item_idx.shape[0]
            if n_u == 0:
                logger.debug("no ratings for user %d", i)
                continue
            item_features = self.item_features_.take(item_idx, axis=0)
            ratings = self.ratings_csr_[i, :].data - self.mean_rating_

            A_i = (np.dot(item_features.T, item_features) +
                   self.reg * n_u * np.eye(self.n_feature))
            V_i = np.dot(item_features.T, ratings)
            self.user_features_[i, :] = np.dot(inv(A_i), V_i)


    def fit(self, ratings, n_iters=50):

        check_ratings(ratings, self.n_user, self.n_item,
                      self.max_rating, self.min_rating)
        self.mean_rating_ = np.mean(ratings.take(2, axis=1))
        self.ratings_csr_ = build_user_item_matrix(
            self.n_user, self.n_item, ratings)
        self.ratings_csc_ = self.ratings_csr_.tocsc()

        last_rmse = None
        start = time.time()
        for iteration in xrange(n_iters):
            logger.debug("iteration %d...", iteration)

            self._update_user_feature()
            self._update_item_feature()

            train_predictions = self.predict(ratings.take([0, 1], axis=1))
            train_rmse = RMSE(train_predictions, ratings.take(2, axis=1))
            end = time.time() - start
            logger.info("iter: %d, train RMSE: %.6f, time: %.6f, size: %d", iteration, train_rmse, end, self.n_size)

    def predict(self, data):

        if not self.mean_rating_:
            raise NotFittedError("Please fit model before run predict")

        u_features = self.user_features_.take(data.take(0, axis=1), axis=0)
        i_features = self.item_features_.take(data.take(1, axis=1), axis=0)
        predictions = np.sum(u_features * i_features, 1) + self.mean_rating_

        if self.max_rating:
            predictions[predictions > self.max_rating] = self.max_rating

        if self.min_rating:
            predictions[predictions < self.min_rating] = self.min_rating
        return predictions
