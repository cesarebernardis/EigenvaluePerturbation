#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: nonecheck=False
#cython: cdivision=True
#cython: unpack_method_calls=True
#cython: overflowcheck=False

#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

from RecSysFramework.Recommender.Utils import check_matrix

import cython
import sys
import time

import numpy as np
cimport numpy as np


cdef class OptimalEigenPerturbationHelper:

    cdef int n_users, n_items, n_factors
    cdef int[:] urm_indptr, urm_indices

    cdef double[:] urm_data, norms

    def __init__(self, URM_train):

        URM_train = check_matrix(URM_train, 'csr')
        URM_train = URM_train.sorted_indices()

        self.n_users, self.n_items = URM_train.shape

        self.urm_indices = URM_train.indices
        self.urm_data = np.array(URM_train.data, dtype=np.float64)
        self.urm_indptr = URM_train.indptr


    def compute_similarity_matrix(self, l2=0.1, shrink=10.0):

        cdef int u, i, j
        cdef int item_i, item_j
        cdef double value, norm, cl2, cshrink
        cdef long start_time_epoch = time.time()
        cdef long last_print_time = start_time_epoch

        cdef double[:,:] similarity_matrix_view

        cl2 = l2
        cshrink = shrink

        similarity_matrix = np.zeros((self.n_items, self.n_items), dtype=np.float64)
        similarity_matrix_view = similarity_matrix

        for u in range(self.n_users):

            norm = 0.
            for i in range(self.urm_indptr[u], self.urm_indptr[u+1]):
                norm += self.urm_data[i] * self.urm_data[i]
            norm = cl2 * 2 *norm + cshrink

            for i in range(self.urm_indptr[u], self.urm_indptr[u+1]):
                item_i = self.urm_indices[i]
                for j in range(i+1, self.urm_indptr[u+1]):
                    item_j = self.urm_indices[j]
                    value = self.urm_data[i] * self.urm_data[j] / norm
                    similarity_matrix_view[item_i, item_j] += value
                    similarity_matrix_view[item_j, item_i] += value

            if time.time() - last_print_time > 30 or u == self.n_users-1:
                print("OptimalEigenPerturbation: Processed {} ( {:.2f}% ) in {:.2f} seconds. Items per second: {:.0f}"
                      .format(u+1, 100.0 * (u+1) / self.n_users,
                              time.time() - start_time_epoch,
                              float(u + 1) / (time.time() - start_time_epoch))
                )

                last_print_time = time.time()
                sys.stdout.flush()
                sys.stderr.flush()

        for i in range(self.n_items):
            similarity_matrix_view[i, i] = 0.0

        return similarity_matrix
