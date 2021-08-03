#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/06/18

@author: Anonymous Author
"""

from RecSysFramework.Recommender import BaseMatrixFactorizationRecommender, ItemSimilarityMatrixRecommender

from sklearn.utils.extmath import randomized_svd

import similaripy as sim
import scipy.sparse as sps


class PureSVD(BaseMatrixFactorizationRecommender):
    """ PureSVDRecommender"""

    RECOMMENDER_NAME = "PureSVD"

    def __init__(self, URM_train):
        super(PureSVD, self).__init__(URM_train)


    def fit(self, num_factors=100, random_seed=None):

        print(self.RECOMMENDER_NAME + ": Computing SVD decomposition...")

        U, Sigma, VT = randomized_svd(self.URM_train,
                                      n_components=num_factors,
                                      random_state=random_seed)

        s_Vt = sps.diags(Sigma)*VT

        self.USER_factors = U
        self.ITEM_factors = s_Vt.T

        print(self.RECOMMENDER_NAME + ": Computing SVD decomposition... Done!")



class PureSVDSimilarity(ItemSimilarityMatrixRecommender):
    """ PureSVDSimilarityRecommender"""

    RECOMMENDER_NAME = "PureSVDSimilarity"

    def __init__(self, URM_train, sparse_weights=True):
        super(PureSVDSimilarity, self).__init__(URM_train)
        self.sparse_weights = sparse_weights


    def fit(self, num_factors=100, topK=100, random_seed=None):

        print(self.RECOMMENDER_NAME + ": Computing SVD decomposition...")

        U, Sigma, VT = randomized_svd(self.URM_train,
                                      n_components=num_factors,
                                      random_state=random_seed)

        print(self.RECOMMENDER_NAME + ": Computing SVD decomposition... Done!")

        self.W_sparse = sim.dot_product(sps.csr_matrix(VT.T),
                                        sps.csr_matrix(sps.diags(Sigma)*VT),
                                        shrink=0.0, k=topK).transpose().tocsr()
