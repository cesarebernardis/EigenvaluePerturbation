#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 30/11/18

@author: Anonymous Author
"""

import numpy as np
import scipy.sparse as sps

from RecSysFramework.DataManager import Dataset
from RecSysFramework.Utils import IncrementalSparseMatrix

from .DataSplitter import DataSplitter



class LeaveKOut(DataSplitter):
    """
    The splitter tries to load from the specific folder related to a dataset, a split in the format corresponding to
    the splitter class. Basically each split is in a different subfolder
    - The "original" subfolder contains the whole dataset, is composed by a single URM with all data and may contain
        ICMs as well, either one or many, depending on the dataset
    - The other subfolders "warm", "cold" ecc contains the splitted data.

    The dataReader class involvement is limited to the following cased:
    - At first the dataSplitter tries to load from the subfolder corresponding to that split. Say "warm"
    - If the dataReader is succesful in loading the files, then a split already exists and the loading is complete
    - If the dataReader raises a FileNotFoundException, then no split is available.
    - The dataSplitter then creates a new instance of dataReader using default parameters, so that the original data will be loaded
    - At this point the chosen dataSplitter takes the URM_all and selected ICM to perform the split
    - The dataSplitter saves the splitted data in the appropriate subfolder.
    - Finally, the dataReader is instantiated again with the correct parameters, to load the data just saved
    """

    def __init__(self, k_value=1, forbid_new_split=False, force_new_split=False, with_validation=True, random_seed=42):
        """

        :param dataReader_object:
        :param n_folds:
        :param force_new_split:
        :param forbid_new_split:
        :param save_folder_path:    path in which to save the loaded dataset
                                    None    use default "dataset_name/split_name/"
                                    False   do not save
        """

        assert k_value >= 1, "DataSplitterLeaveKOut: k_value must be greater or equal to 1"
        self.k_value = k_value

        super(LeaveKOut, self).__init__(forbid_new_split=forbid_new_split, force_new_split=force_new_split,
                                        allow_cold_users=False, with_validation=with_validation,
                                        random_seed=random_seed)

    def get_name(self):
        return "leave_{}_out{}".format(self.k_value, "" if self.allow_cold_users else "_no_cold_users")


    def split(self, dataset):

        super(LeaveKOut, self).split(dataset)

        URM = sps.csr_matrix(dataset.get_URM())

        split_number = 2
        if self.with_validation:
            split_number += 1

        # Min interactions at least self.k_value for each split +1 for train
        min_user_interactions = split_number * (self.k_value - 1) + 1

        users_to_preserve = np.arange(URM.shape[0])
        if not self.allow_cold_users:
            user_interactions = np.ediff1d(URM.indptr)
            users_to_preserve = users_to_preserve[user_interactions >= min_user_interactions]

            print("DataSplitterLeaveKOut: Removing {} of {} users because they have less "
                  "than the {} interactions required for {} splits"
                  .format(URM.shape[0] - len(users_to_preserve), URM.shape[0], min_user_interactions, split_number))
        users_to_remove = np.setdiff1d(np.arange(URM.shape[0]), users_to_preserve)

        n_users, n_items = URM.shape
        user_indices = []
        URM_train, URM_test, URM_validation = {}, {}, {}

        #Select apriori how to randomizely sort every user
        for user_id in users_to_preserve.tolist():
            user_indices.append(np.random.permutation(URM.indptr[user_id + 1] - URM.indptr[user_id]))

        for URM_name in dataset.get_URM_names():

            URM = dataset.get_URM(URM_name).tocsr()

            URM_train_builder = IncrementalSparseMatrix(auto_create_row_mapper=False, n_rows=n_users,
                                                        auto_create_col_mapper=False, n_cols=n_items)

            URM_test_builder = IncrementalSparseMatrix(auto_create_row_mapper=False, n_rows=n_users,
                                                       auto_create_col_mapper=False, n_cols=n_items)

            if self.with_validation:
                URM_validation_builder = IncrementalSparseMatrix(auto_create_row_mapper=False, n_rows=n_users,
                                                                 auto_create_col_mapper=False, n_cols=n_items)

            for i, user_id in enumerate(users_to_preserve.tolist()):
                start_user_position = URM.indptr[user_id]
                end_user_position = URM.indptr[user_id + 1]

                user_profile = URM.indices[start_user_position:end_user_position]
                indices_to_suffle = user_indices[i]

                user_interaction_items = user_profile[indices_to_suffle]
                user_interaction_data = URM.data[start_user_position:end_user_position][indices_to_suffle]

                # Test interactions
                user_interaction_items_test = user_interaction_items[:self.k_value]
                user_interaction_data_test = user_interaction_data[:self.k_value]

                URM_test_builder.add_data_lists([user_id] * self.k_value,
                                                user_interaction_items_test,
                                                user_interaction_data_test)

                train_start = self.k_value
                # validation interactions
                if self.with_validation:
                    user_interaction_items_validation = user_interaction_items[self.k_value:self.k_value * 2]
                    user_interaction_data_validation = user_interaction_data[self.k_value:self.k_value * 2]

                    URM_validation_builder.add_data_lists([user_id] * self.k_value,
                                                          user_interaction_items_validation,
                                                          user_interaction_data_validation)
                    train_start = self.k_value * 2

                # Train interactions
                user_interaction_items_train = user_interaction_items[train_start:]
                user_interaction_data_train = user_interaction_data[train_start:]

                URM_train_builder.add_data_lists([user_id] * len(user_interaction_items_train),
                                                 user_interaction_items_train, user_interaction_data_train)

            URM_train[URM_name] = URM_train_builder.get_SparseMatrix()
            URM_test[URM_name] = URM_test_builder.get_SparseMatrix()

            if self.with_validation:
                URM_validation[URM_name] = URM_validation_builder.get_SparseMatrix()

        train = Dataset(dataset.get_name(), base_folder=dataset.get_base_folder(),
                        postprocessings=dataset.get_postprocessings(),
                        URM_dict=URM_train, URM_mappers_dict=dataset.get_URM_mappers_dict(),
                        ICM_dict=dataset.get_ICM_dict(), ICM_mappers_dict=dataset.get_ICM_mappers_dict(),
                        UCM_dict=dataset.get_UCM_dict(), UCM_mappers_dict=dataset.get_UCM_mappers_dict())
        train.remove_users(users_to_remove)

        test = Dataset(dataset.get_name(), base_folder=dataset.get_base_folder(),
                       postprocessings=dataset.get_postprocessings(),
                       URM_dict=URM_test, URM_mappers_dict=dataset.get_URM_mappers_dict(),
                       ICM_dict=dataset.get_ICM_dict(), ICM_mappers_dict=dataset.get_ICM_mappers_dict(),
                       UCM_dict=dataset.get_UCM_dict(), UCM_mappers_dict=dataset.get_UCM_mappers_dict())
        test.remove_users(users_to_remove)

        if self.with_validation:
            validation = Dataset(dataset.get_name(), base_folder=dataset.get_base_folder(),
                                 postprocessings=dataset.get_postprocessings(),
                                 URM_dict=URM_validation, URM_mappers_dict=dataset.get_URM_mappers_dict(),
                                 ICM_dict=dataset.get_ICM_dict(), ICM_mappers_dict=dataset.get_ICM_mappers_dict(),
                                 UCM_dict=dataset.get_UCM_dict(), UCM_mappers_dict=dataset.get_UCM_mappers_dict())
            validation.remove_users(users_to_remove)
            return train, test, validation
        else:
            return train, test

