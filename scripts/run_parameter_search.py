#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

from RecSysFramework.ExperimentalConfig import EXPERIMENTAL_CONFIG
from RecSysFramework.Recommender.DataIO import DataIO
from RecSysFramework.Recommender.KNN import EigenPerturbation

from RecSysFramework.Evaluation import EvaluatorHoldout

from RecSysFramework.ParameterTuning.Utils import run_parameter_search


def read_data_split_and_search():

    cutoffs = EXPERIMENTAL_CONFIG['cutoffs']

    results_path = "./"
    results_filename = results_path + "optimized_results.csv"

    for splitter in EXPERIMENTAL_CONFIG['splits']:

        for dataset_config in EXPERIMENTAL_CONFIG['datasets']:
            datareader = dataset_config['datareader']()
            postprocessings = dataset_config['postprocessings']

            for algorithm in EXPERIMENTAL_CONFIG["item-based-algorithms"]:

                train, test, validation = splitter.load_split(datareader, postprocessings=postprocessings)

                input_folder_path = train.get_complete_folder() + \
                                    os.path.join(splitter.get_name(), algorithm.RECOMMENDER_NAME) + \
                                    os.sep

                filename = algorithm.RECOMMENDER_NAME + "_metadata"

                # Without perturbation
                run_parameter_search(
                    algorithm, splitter.get_name(), train, validation, dataset_test=test,
                    metric_to_optimize="RECALL", cutoff_to_optimize=10, resume_from_saved=True,
                    n_cases=50, n_random_starts=15, save_model="no"
                )

                # With perturbation
                run_parameter_search(
                    algorithm, splitter.get_name(), train, validation,
                    dataset_test=test, output_folder_path=input_folder_path,
                    metric_to_optimize="RECALL", cutoff_to_optimize=10, resume_from_saved=True,
                    n_cases=50, n_random_starts=15, is_eigenperturbation=True, save_model="no"
                )






if __name__ == '__main__':

    read_data_split_and_search()
