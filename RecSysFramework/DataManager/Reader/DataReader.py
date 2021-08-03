#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 01/01/2018

@author: Anonymous Author
"""


import scipy.sparse as sps
import numpy as np
import pickle, os

from RecSysFramework.DataManager import Dataset


#################################################################################################################
#############################
#############################               DATA READER
#############################
#################################################################################################################


class DataReader(object):
    """
    Abstract class for the DataReaders, each shoud be implemented for a specific dataset
    Reader has the following functions:
     - It loads the data of the original dataset and saves it into sparse matrices
     - It exposes the following functions
        - load_data(save_folder_path=None)        loads the data and saves into the specified folder, if None uses default
        - get_URM_all()                             returns a copy of the whole URM
        - get_ICM_from_name(ICM_name)               returns a copy of the specified ICM
        - get_loaded_ICM_names()                    returns a copy of the loaded ICM names, which can be used in get_ICM_from_name
        - get_loaded_ICM_dict()                     returns a copy of the loaded ICM in a dictionary [ICM_name]->ICM_sparse
        - DATASET_SUBFOLDER_DEFAULT                 path of the data folder
        - item_original_ID_to_index
        - user_original_ID_to_index

    """
    DATASET_SPLIT_ROOT_FOLDER = "data" + os.sep
    DATASET_OFFLINE_ROOT_FOLDER = "data" + os.sep

    # Available URM split
    AVAILABLE_URM = ["URM_all"]

    # Available ICM for the given dataset, there might be no ICM, one or many
    AVAILABLE_ICM = []

    # Available ICM for the given dataset, there might be no ICM, one or many
    AVAILABLE_UCM = []

    # This flag specifies if the given dataset contains implicit preferences or explicit ratings
    MANDATORY_POSTPROCESSINGS = []


    def __init__(self, reload_from_original_data=False):

        super(DataReader, self).__init__()

        self.reload_from_original_data = reload_from_original_data
        if self.reload_from_original_data:
            print("Reader: reload_from_original_data is True, previously loaded data will be ignored")


    def get_dataset_name(self):
        return self._get_dataset_name_root()[:-1]


    def _get_dataset_name_root(self):
        """
        Returns the root of the folder tree which contains all of the dataset data/splits and files

        :return: Dataset_name/
        """
        raise NotImplementedError("Reader: The following method was not implemented for the required dataset. Impossible to load the data")


    def get_default_save_path(self):
        return self.DATASET_SPLIT_ROOT_FOLDER + self._get_dataset_name_root()


    def get_complete_default_save_path(self, postprocessings=None):

        if postprocessings is None:
            postprocessings = self.MANDATORY_POSTPROCESSINGS
        else:
            postprocessings = self.MANDATORY_POSTPROCESSINGS + postprocessings

        savepath = self.get_default_save_path()
        if postprocessings:
            savepath += os.path.join(*[p.get_name() for p in postprocessings]) + os.sep

        return savepath


    def load_data(self, save_folder_path=None, postprocessings=None):
        """

        :param save_folder_path:    path in which to save the loaded dataset
                                    None    use default "dataset_name/original/"
                                    False   do not save
        :return:
        """

        postprocessing_index = 0
        if postprocessings is None:
            postprocessings = self.MANDATORY_POSTPROCESSINGS
        else:
            postprocessings = self.MANDATORY_POSTPROCESSINGS + postprocessings

        progressive_save_folder_path = None
        if save_folder_path is None:
            # Check if partial preprocessings have already been saved and load them
            save_folder_path = self.get_default_save_path()
            progressive_save_folder_path = save_folder_path
            for postprocessing in postprocessings:
                if os.path.exists(progressive_save_folder_path + postprocessing.get_name()):
                    progressive_save_folder_path += postprocessing.get_name() + os.sep
                    postprocessing_index += 1
                else:
                    break

            fallback_steps = 0
            while self.all_files_available(save_folder_path + os.sep.join(p.get_name()
                                                for p in postprocessings[:postprocessing_index-fallback_steps])) \
                    and fallback_steps > postprocessing_index:
                fallback_steps += 1

            postprocessing_index -= fallback_steps
            progressive_save_folder_path = save_folder_path + os.sep.join(p.get_name()
                                                    for p in postprocessings[:postprocessing_index]) + os.sep

        dataset = None
        # If save_folder_path contains any path try to load a previously built split from it
        if save_folder_path is not None and not self.reload_from_original_data:

            # save folder path given
            if progressive_save_folder_path is None:
                progressive_save_folder_path = save_folder_path

            try:
                urm, urm_mappers, icm, icm_mappers, ucm, ucm_mappers = self.load_from_saved_sparse_matrix(progressive_save_folder_path)
                dataset = Dataset(self.get_dataset_name(), base_folder=self.get_default_save_path(),
                                  postprocessings=postprocessings[:postprocessing_index],
                                  URM_dict=urm, URM_mappers_dict=urm_mappers,
                                  ICM_dict=icm, ICM_mappers_dict=icm_mappers,
                                  UCM_dict=ucm, UCM_mappers_dict=ucm_mappers)

            except Exception as e:
                postprocessing_index = 0
                print("Reader: Preloaded data not found or corrupted ({}), reading from original files...".format(e))
                pass

        if dataset is None:
            dataset = self._load_from_original_file()

        for postprocessing in postprocessings[postprocessing_index:]:
            dataset = postprocessing.apply(dataset)

        dataset.print_statistics()

        return dataset


    def _load_from_original_file(self):
        raise NotImplementedError("Reader: The following method was not implemented for the required dataset. Impossible to load the data")


    def all_files_available(self, save_folder_path, filename_suffix=""):

        for filename in self.AVAILABLE_UCM + self.AVAILABLE_ICM + self.AVAILABLE_URM:
            if not os.path.isfile(save_folder_path + filename + filename_suffix + ".npz") or \
                    not os.path.isfile(save_folder_path + filename + filename_suffix + "_mapper"):
                return False

        return True
        

    def load_from_saved_sparse_matrix(self, save_folder_path, filename_suffix=""):

        icm = {}
        icm_mappers = {}
        for filename in self.AVAILABLE_ICM:
            print("Reader: Loading {}...".format(save_folder_path + filename + filename_suffix))
            icm[filename] = sps.load_npz("{}.npz".format(save_folder_path + filename + filename_suffix))
            with open("{}_mapper".format(save_folder_path + filename + filename_suffix), "rb") as file:
                icm_mappers[filename] = pickle.load(file)

        ucm = {}
        ucm_mappers = {}
        for filename in self.AVAILABLE_UCM:
            print("Reader: Loading {}...".format(save_folder_path + filename + filename_suffix))
            ucm[filename] = sps.load_npz("{}.npz".format(save_folder_path + filename + filename_suffix))
            with open("{}_mapper".format(save_folder_path + filename + filename_suffix), "rb") as file:
                ucm_mappers[filename] = pickle.load(file)

        urm = {}
        urm_mappers = {}
        for filename in self.AVAILABLE_URM:
            print("Reader: Loading {}...".format(save_folder_path + filename + filename_suffix))
            urm[filename] = sps.load_npz("{}.npz".format(save_folder_path + filename + filename_suffix))
            with open("{}_mapper".format(save_folder_path + filename + filename_suffix), "rb") as file:
                urm_mappers[filename] = pickle.load(file)

        print("Reader: Loading complete!")

        return urm, urm_mappers, icm, icm_mappers, ucm, ucm_mappers


    def _merge_ICM(self, ICM1, ICM2, mapper_ICM1, mapper_ICM2):

        assert ICM1.shape[0] == ICM2.shape[0], "Wrong shapes of ICMs: {} - {}".format(ICM1.shape, ICM2.shape)
        ICM_all = sps.hstack([ICM1, ICM2], format='csr')

        mapper_ICM_all = mapper_ICM1.copy()

        for key in mapper_ICM2.keys():
            mapper_ICM_all[key] = mapper_ICM2[key] + len(mapper_ICM1)

        return  ICM_all, mapper_ICM_all


