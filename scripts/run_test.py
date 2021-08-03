#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

from RecSysFramework.ExperimentalConfig import EXPERIMENTAL_CONFIG
from RecSysFramework.Recommender.DataIO import DataIO
from RecSysFramework.Recommender.KNN import EigenPerturbation

from RecSysFramework.Evaluation import EvaluatorHoldout

from RecSysFramework.ParameterTuning.Utils import run_parameter_search


def find_best_hp(folder_path, filename):

    best_hp = {}
    for splitter in EXPERIMENTAL_CONFIG['splits']:
        for dataset_config in EXPERIMENTAL_CONFIG['datasets']:
            datareader = dataset_config['datareader']()
            postprocessings = dataset_config['postprocessings']
            dataset_name = datareader.get_dataset_name()

            train, test, validation = splitter.load_split(datareader, postprocessings=postprocessings)
            best_hp[dataset_name] = {}

            for algorithm in EXPERIMENTAL_CONFIG["item-based-algorithms"]:
                recommender_name = algorithm.RECOMMENDER_NAME
                best_hp[dataset_name][recommender_name] = {}

                input_folder_path = train.get_complete_folder() + \
                                    os.path.join(splitter.get_name(), recommender_name) + \
                                    os.sep

                data_filename = recommender_name + "_metadata"

                try:
                    dataIO = DataIO(folder_path=input_folder_path)
                    data_dict = dataIO.load_data(file_name=data_filename)
                    best_hp[dataset_name][recommender_name]['0'] = data_dict["hyperparameters_best"]

                    dataIO = DataIO(folder_path=input_folder_path[:-1] + "-EigenPerturbation" + os.sep)
                    data_dict = dataIO.load_data(file_name=data_filename)
                    best_hp[dataset_name][recommender_name]['1'] = data_dict["hyperparameters_best"]
                except FileNotFoundError:
                    raise Exception("Best hyperparameters not found! Run parameter search.")

    dataIO = DataIO(folder_path=folder_path)
    dataIO.save_data(filename, best_hp)
    return best_hp


def read_data_split_and_search():

    cutoffs = EXPERIMENTAL_CONFIG['cutoffs']

    results_path = "./"
    results_filename = results_path + "optimized_results.csv"

    if not os.path.isfile(results_filename):
        with open(results_filename, "w") as file:
            print("splitter,fold,dataset,recommender,perturbation,cutoff," +
                  ','.join(m.value.METRIC_NAME for m in EXPERIMENTAL_CONFIG['recap_metrics']),
                  file=file)

    folder_path = "./data/"
    filename = "best-hyperparameters"
    if not os.path.isfile(folder_path + filename + ".zip"):
        best_hp = find_best_hp(folder_path, filename)
    else:
        dataIO = DataIO(folder_path=folder_path)
        best_hp = dataIO.load_data(file_name=filename)

    for splitter in EXPERIMENTAL_CONFIG['splits']:

        for dataset_config in EXPERIMENTAL_CONFIG['datasets']:

            datareader = dataset_config['datareader']()
            postprocessings = dataset_config['postprocessings']
            dataset_name = datareader.get_dataset_name()

            for algorithm in EXPERIMENTAL_CONFIG["item-based-algorithms"]:

                recommender_name = algorithm.RECOMMENDER_NAME
                params = best_hp[dataset_name][recommender_name]

                for fold in range(EXPERIMENTAL_CONFIG['n_folds']):

                    train, test, validation = splitter.load_split(datareader, postprocessings=postprocessings,
                                                                  filename_suffix="_{}".format(fold))

                    URM_train = train.get_URM() + validation.get_URM()
                    URM_test = test.get_URM()

                    evaluator = EvaluatorHoldout(cutoffs)
                    evaluator.global_setup(URM_test)

                    recommender = algorithm(URM_train)
                    recommender.fit(**params['0'])

                    metrics_handler = evaluator.evaluateRecommender(recommender)
                    results = metrics_handler.get_results_dictionary()

                    with open(results_filename, "a") as file:
                        for cutoff in cutoffs:
                            print("{},{},{},{},{},{},{}".format(splitter.get_name(), fold,
                                     datareader.get_dataset_name(), recommender_name, 0, cutoff,
                                     ','.join("{:.5f}".format(results[cutoff][m.value.METRIC_NAME])
                                              for m in EXPERIMENTAL_CONFIG['recap_metrics'])),
                                  file=file)

                    recommender = algorithm(URM_train)
                    recommender.fit(**params['1'])

                    eigen_recommender = EigenPerturbation(URM_train, recommender.get_W_sparse())
                    eigen_recommender.fit(perturbation=1.0)
                    recommendername = recommender.RECOMMENDER_NAME
                    del (recommender)

                    metrics_handler = evaluator.evaluateRecommender(eigen_recommender)
                    del (eigen_recommender)
                    results = metrics_handler.get_results_dictionary()

                    with open(results_filename, "a") as file:
                        for cutoff in cutoffs:
                            print("{},{},{},{},{},{},{}".format(splitter.get_name(), fold,
                                    datareader.get_dataset_name(), recommendername, 1.0, cutoff,
                                    ','.join("{:.5f}".format(results[cutoff][m.value.METRIC_NAME])
                                             for m in EXPERIMENTAL_CONFIG['recap_metrics'])),
                                  file=file)




if __name__ == '__main__':

    read_data_split_and_search()
