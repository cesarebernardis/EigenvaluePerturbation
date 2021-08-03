#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/12/18

@author: Anonymous Author
"""

import time
import os
import numpy as np

from RecSysFramework.Recommender.KNN import EigenPerturbation

from RecSysFramework.Evaluation.Evaluator import get_result_string
from RecSysFramework.ParameterTuning.SearchBayesianSkopt import SearchBayesianSkopt, get_result_string_evaluate_on_validation

def _compute_avg_time_non_none_values(data_list):

    non_none_values = sum([value is not None for value in data_list])
    total_value = sum([value if value is not None else 0.0 for value in data_list])

    return total_value, total_value/non_none_values



class SearchBayesianSkoptEigenPerturbationBase(SearchBayesianSkopt):

    ALGORITHM_NAME = "SearchBayesianSkoptEigenPerturbationBase"

    def __init__(self, recommender_class, eigenperturbation_class, evaluator_validation=None, evaluator_test=None):
        self.eigenperturbation_class = eigenperturbation_class
        super(SearchBayesianSkoptEigenPerturbationBase, self).__init__(recommender_class,
                                                  evaluator_validation=evaluator_validation,
                                                  evaluator_test=evaluator_test)

    def search(self, recommender_input_args,
               parameter_search_space,
               metric_to_optimize="MAP",
               n_cases=20,
               n_random_starts=5,
               output_folder_path=None,
               output_file_name_root=None,
               save_model="best",
               save_metadata=True,
               resume_from_saved=False,
               recommender_input_args_last_test=None,
               ):

        super(SearchBayesianSkoptEigenPerturbationBase, self).search(
            recommender_input_args, parameter_search_space,
            metric_to_optimize=metric_to_optimize,
            n_cases=n_cases,
            n_random_starts=n_random_starts,
            output_folder_path=output_folder_path[:-1] + "-" + self.eigenperturbation_class.RECOMMENDER_NAME + os.sep,
            output_file_name_root=output_file_name_root,
            save_model=save_model,
            save_metadata=save_metadata,
            resume_from_saved=resume_from_saved,
            recommender_input_args_last_test=recommender_input_args_last_test
        )


    def _evaluate_on_validation(self, current_fit_parameters):

        start_time = time.time()

        # Construct a new recommender instance
        recommender_instance = self.recommender_class(*self.recommender_input_args.CONSTRUCTOR_POSITIONAL_ARGS,
                                                      **self.recommender_input_args.CONSTRUCTOR_KEYWORD_ARGS)

        print("{}: Testing config:".format(self.ALGORITHM_NAME), current_fit_parameters)

        recommender_instance.fit(*self.recommender_input_args.FIT_POSITIONAL_ARGS,
                                 **self.recommender_input_args.FIT_KEYWORD_ARGS,
                                 **current_fit_parameters)

        eprecommender = self.eigenperturbation_class(self.recommender_input_args.CONSTRUCTOR_POSITIONAL_ARGS[0],
                                          recommender_instance.get_W_sparse())

        eprecommender.fit(perturbation=1.0)

        train_time = time.time() - start_time
        start_time = time.time()

        # Evaluate recommender and get results for the first cutoff
        metrics_handler = self.evaluator_validation.evaluateRecommender(eprecommender)
        result_dict = metrics_handler.get_results_dictionary(use_metric_name=False)
        result_dict = result_dict[list(result_dict.keys())[0]]

        evaluation_time = time.time() - start_time

        result_string = get_result_string_evaluate_on_validation(result_dict, n_decimals=7)

        return result_dict, result_string, eprecommender, train_time, evaluation_time


class SearchBayesianSkoptEigenPerturbation(SearchBayesianSkoptEigenPerturbationBase):

    ALGORITHM_NAME = "SearchBayesianSkoptEigenPerturbation"

    def __init__(self, recommender_class, evaluator_validation=None, evaluator_test=None):
        super(SearchBayesianSkoptEigenPerturbation, self).__init__(
            recommender_class, EigenPerturbation,
            evaluator_validation=evaluator_validation,
            evaluator_test=evaluator_test
        )

