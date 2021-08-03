#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 17/07/19

@author: Anonymous Author
"""

import numpy as np
import scipy.sparse as sps
import similaripy as sim

from RecSysFramework.Recommender import ItemSimilarityMatrixRecommender
from RecSysFramework.Recommender.Utils import check_matrix


class EigenPerturbation(ItemSimilarityMatrixRecommender):

    """ EigenPerturbation recommender"""

    RECOMMENDER_NAME = "EigenPerturbation"


    def __init__(self, URM_train, S):

        super(EigenPerturbation, self).__init__(URM_train)
        self.S = check_matrix(S, 'csr')


    def fit(self, perturbation=None, topK=None):
        self.topK = topK
        if perturbation is None:
            self.perturbation = self.URM_train.max()
        else:
            self.perturbation = perturbation
        self.S.setdiag(0)
        self.W_sparse = (self.S + self.S.transpose()).tocsr()
        if topK is not None:
            self.W_sparse = sim.dot_product(self.W_sparse, sps.eye(self.W_sparse.shape[0]), k=self.topK).transpose().tocsr()

