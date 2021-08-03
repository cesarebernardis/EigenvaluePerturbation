#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 26/06/18

@author: Anonymous Author, Anonymous Author
"""

import numpy as np
import scipy.sparse as sps
import time, sys, copy

from enum import Enum
from RecSysFramework.Utils import seconds_to_biggest_unit

from .Metrics import *



class EvaluatorMetrics(Enum):

    ROC_AUC = ROC_AUC
    PRECISION = Precision
    PRECISION_MIN_TEST_LEN = PrecisionMinTestLen
    RECALL_MIN_TEST_LEN = RecallMinTestLen
    RECALL = Recall
    MAP = MAP
    MRR = MRR
    NDCG = NDCG
    F1 = F1
    HIT_RATE = HIT_RATE
    ARHR = ARHR
    RMSE = RMSE
    NOVELTY = Novelty
    AVERAGE_POPULARITY = AveragePopularity
    DIVERSITY_MEAN_INTER_LIST = Diversity_MeanInterList
    DIVERSITY_HERFINDAHL = Diversity_Herfindahl
    COVERAGE_ITEM = CoverageItem
    COVERAGE_ITEM_TEST = Coverage_Test_Item
    COVERAGE_USER = CoverageUser
    DIVERSITY_GINI = Gini_Diversity
    SHANNON_ENTROPY = Shannon_Entropy

    RATIO_DIVERSITY_HERFINDAHL = Ratio_Diversity_Herfindahl
    RATIO_DIVERSITY_GINI = Ratio_Diversity_Gini
    RATIO_SHANNON_ENTROPY = Ratio_Shannon_Entropy
    RATIO_AVERAGE_POPULARITY = Ratio_AveragePopularity
    RATIO_NOVELTY = Ratio_Novelty



def create_empty_metrics_dict(n_items, n_users, URM_train, ignore_items, ignore_users, cutoff, diversity_similarity_object):

    empty_dict = {}

    for metric in EvaluatorMetrics:
        if metric == EvaluatorMetrics.COVERAGE_ITEM:
            empty_dict[metric.value] = CoverageItem(n_items, ignore_items)

        elif metric == EvaluatorMetrics.COVERAGE_ITEM_TEST:
            empty_dict[metric.value] = Coverage_Test_Item(n_items, ignore_items)

        elif metric == EvaluatorMetrics.DIVERSITY_GINI:
            empty_dict[metric.value] = Gini_Diversity(n_items, ignore_items)

        elif metric == EvaluatorMetrics.SHANNON_ENTROPY:
            empty_dict[metric.value] = Shannon_Entropy(n_items, ignore_items)

        elif metric == EvaluatorMetrics.COVERAGE_USER:
            empty_dict[metric.value] = CoverageUser(n_users, ignore_users)

        elif metric == EvaluatorMetrics.DIVERSITY_MEAN_INTER_LIST:
            empty_dict[metric.value] = Diversity_MeanInterList(n_items, cutoff)

        elif metric == EvaluatorMetrics.DIVERSITY_HERFINDAHL:
            empty_dict[metric.value] = Diversity_Herfindahl(n_items, ignore_items)

        elif metric == EvaluatorMetrics.NOVELTY:
            empty_dict[metric.value] = Novelty(URM_train)

        elif metric == EvaluatorMetrics.AVERAGE_POPULARITY:
            empty_dict[metric.value] = AveragePopularity(URM_train)

        elif metric == EvaluatorMetrics.MAP:
            empty_dict[metric.value] = MAP()

        elif metric == EvaluatorMetrics.MRR:
            empty_dict[metric.value] = MRR()

        elif metric == EvaluatorMetrics.RATIO_DIVERSITY_GINI:
            empty_dict[metric.value] = Ratio_Diversity_Gini(URM_train, ignore_users)

        elif metric == EvaluatorMetrics.RATIO_DIVERSITY_HERFINDAHL:
            empty_dict[metric.value] = Ratio_Diversity_Herfindahl(URM_train, ignore_users)

        elif metric == EvaluatorMetrics.RATIO_SHANNON_ENTROPY:
            empty_dict[metric.value] = Ratio_Shannon_Entropy(URM_train, ignore_users)

        elif metric == EvaluatorMetrics.RATIO_AVERAGE_POPULARITY:
            empty_dict[metric.value] = Ratio_AveragePopularity(URM_train)

        elif metric == EvaluatorMetrics.RATIO_NOVELTY:
            empty_dict[metric.value] = Ratio_Novelty(URM_train)

        elif metric == EvaluatorMetrics.DIVERSITY_SIMILARITY:
                if diversity_similarity_object is not None:
                    empty_dict[metric.value] = copy.deepcopy(diversity_similarity_object)
        else:
            empty_dict[metric.value] = 0.0

    return empty_dict



def get_result_string(results_run, n_decimals=7):

    output_str = ""

    for cutoff in results_run.keys():

        results_run_current_cutoff = results_run[cutoff]

        output_str += "CUTOFF: {} - ".format(cutoff)

        for metric in results_run_current_cutoff.keys():
            val = results_run_current_cutoff[metric]
            if isinstance(val, np.ndarray):
                val = val.mean()
            elif isinstance(val, list):
                val = np.mean(val)
            output_str += "{}: {:.{n_decimals}f}, ".format(metric, val, n_decimals=n_decimals)

        output_str += "\n"

    return output_str



class MetricsHandler(object):

    _BASIC_METRICS = [
        EvaluatorMetrics.PRECISION,
        EvaluatorMetrics.RECALL,
        EvaluatorMetrics.F1,
        EvaluatorMetrics.MAP,
        EvaluatorMetrics.NDCG,
        EvaluatorMetrics.ARHR,
        EvaluatorMetrics.COVERAGE_ITEM_TEST,
    ]

    def __init__(self, URM_train, cutoff_list, metrics_list, ignore_users, ignore_items):

        super(MetricsHandler, self).__init__()

        self.n_users, self.n_items = URM_train.shape
        self.item_popularity = np.ediff1d(URM_train.tocsc().indptr)

        self.ignore_users = ignore_users
        self.ignore_items = ignore_items
        self.cutoff_list = cutoff_list.copy()
        self.max_cutoff = max(self.cutoff_list)

        if metrics_list is None:
            metrics_list = self._BASIC_METRICS
        elif not isinstance(metrics_list, list):
            metrics_list = [metrics_list]

        self.metrics = self._init_metric_objects(cutoff_list, metrics_list)
        self.evaluated_users = []


    def _init_metric_objects(self, cutoff_list, metrics_list):

        metrics_objects = {}

        for cutoff in cutoff_list:
            metrics_objects[cutoff] = []
            for metric_enum in metrics_list:
                if isinstance(metric_enum, str):
                    metric_enum = EvaluatorMetrics[metric_enum]
                metric = metric_enum.value
                if metric is CoverageUser:
                    metrics_objects[cutoff].append(CoverageUser(self.n_users, self.ignore_users))
                elif metric is Novelty:
                    metrics_objects[cutoff].append(Novelty(self.item_popularity))
                elif metric is Diversity_MeanInterList:
                    metrics_objects[cutoff].append(Diversity_MeanInterList(self.n_items, cutoff))
                elif issubclass(metric, CumulativeMetric):
                    metrics_objects[cutoff].append(metric())
                else:
                    metrics_objects[cutoff].append(metric(self.n_items, self.ignore_items))

        return metrics_objects


    def add_user_evaluation(self, user_id, recommended_items, predicted_ratings, relevant_items, relevant_items_ratings):

        self.evaluated_users.append(user_id)
        is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

        for cutoff in self.cutoff_list:

            is_relevant_current_cutoff = is_relevant[:cutoff]
            recommended_items_current_cutoff = recommended_items[:cutoff]
            #relevant_items_ratings_current_cutoff = relevant_items_ratings[:cutoff]

            for metric in self.metrics[cutoff]:
                if isinstance(metric, NDCG):
                    metric.add_recommendations(recommended_items_current_cutoff, relevant_items)
                                               #relevance=relevant_items_ratings)
                elif isinstance(metric, RMSE):
                    metric.add_recommendations(predicted_ratings, relevant_items, relevant_items_ratings)
                elif isinstance(metric, CumulativeMetric):
                    metric.add_recommendations(is_relevant_current_cutoff, relevant_items)
                elif isinstance(metric, CoverageUser):
                    metric.add_recommendations(recommended_items_current_cutoff, user_id)
                elif isinstance(metric, Coverage_Test_Item):
                    metric.add_recommendations(recommended_items_current_cutoff, is_relevant_current_cutoff)
                else:
                    metric.add_recommendations(recommended_items_current_cutoff)


    def get_results_dictionary(self, per_user=False, use_metric_name=True):

        results = {}
        for cutoff in self.metrics.keys():
            results[cutoff] = {}
            for metric in self.metrics[cutoff]:
                if use_metric_name:
                    metric_name = metric.METRIC_NAME
                else:
                    metric_name = EvaluatorMetrics(type(metric)).name
                if per_user:
                    try:
                        results[cutoff][metric_name] = metric.get_metric_value_per_user()
                    except:
                        print("MetricsHandler: Metric values per user not available for {}".format(metric_name))
                        pass
                else:
                    results[cutoff][metric_name] = metric.get_metric_value()

        return results


    def get_evaluated_users(self):
        return self.evaluated_users.copy()


    def get_evaluated_users_count(self):
        return len(self.evaluated_users)


    def get_results_string(self):
        return get_result_string(self.get_results_dictionary())



class Evaluator(object):
    """Abstract Evaluator"""

    EVALUATOR_NAME = "Evaluator"

    def __init__(self, cutoff_list, metrics_list=None, minRatingsPerUser=1, exclude_seen=True):

        super(Evaluator, self).__init__()

        if isinstance(cutoff_list, list):
            self.cutoff_list = cutoff_list.copy()
        else:
            self.cutoff_list = [cutoff_list]
        self.max_cutoff = max(self.cutoff_list)

        self.minRatingsPerUser = minRatingsPerUser
        self.exclude_seen = exclude_seen
        self.metrics_list = metrics_list


    def global_setup(self, URM_test, ignore_users=None, ignore_items=None):

        if URM_test is not None:
            self.URM_test = sps.csr_matrix(URM_test)
            self.n_users, self.n_items = URM_test.shape

        if ignore_items is None:
            self.ignore_items_flag = False
            self.ignore_items_ID = np.array([])
        else:
            print("{}: Ignoring {} Items".format(self.EVALUATOR_NAME, len(ignore_items)))
            self.ignore_items_flag = True
            self.ignore_items_ID = np.array(ignore_items)

        # Prune users with an insufficient number of ratings
        # During testing CSR is faster
        numRatings = np.ediff1d(self.URM_test.tocsr().indptr)
        self.usersToEvaluate = np.arange(self.n_users)[numRatings >= self.minRatingsPerUser]

        if ignore_users is not None:
            print("{}: Ignoring {} Users".format(self.EVALUATOR_NAME, len(ignore_users)))
            self.ignore_users_ID = np.array(ignore_users)
        else:
            self.ignore_users_ID = np.array([])

        self.usersToEvaluate = np.setdiff1d(self.usersToEvaluate, self.ignore_users_ID)


    def evaluateRecommender(self, recommender_object, URM_test=None, ignore_users=None, ignore_items=None):
        """
        :param recommender_object: the trained recommender object, a Recommender subclass
        :param URM_test: list of URMs to test the recommender against, or a single URM object
        :param cutoff_list: list of cutoffs to be use to report the scores, or a single cutoff
        """
        self.global_setup(URM_test=URM_test, ignore_users=ignore_users, ignore_items=ignore_items)

        assert self.URM_test is not None, "{}: Test URM not given for evaluation".format(self.EVALUATOR_NAME)


    def get_user_relevant_items(self, user_id):

        assert self.URM_test.getformat() == "csr", \
            "{}: URM_test is not CSR, this will cause errors in getting relevant items".format(self.EVALUATOR_NAME)

        return self.URM_test.indices[self.URM_test.indptr[user_id]:self.URM_test.indptr[user_id+1]]


    def get_user_test_ratings(self, user_id):

        assert self.URM_test.getformat() == "csr", \
            "Evaluator_Base_Class: URM_test is not CSR, this will cause errors in relevant items ratings"

        return self.URM_test.data[self.URM_test.indptr[user_id]:self.URM_test.indptr[user_id+1]]



class EvaluatorHoldout(Evaluator):
    """EvaluatorHoldout"""

    EVALUATOR_NAME = "EvaluatorHoldout"


    def _run_evaluation_on_selected_users(self, recommender_object, block_size=None):

        if block_size is None:
            block_size = min(1000, int(1e8/self.n_items))

        start_time = time.time()
        start_time_print = time.time()

        metrics_handler = MetricsHandler(recommender_object.get_URM_train(), self.cutoff_list, self.metrics_list,
                                         ignore_items=self.ignore_items_ID, ignore_users=self.ignore_users_ID)

        n_users_evaluated = 0

        # Start from -block_size to ensure it to be 0 at the first block
        user_batch_start = 0
        user_batch_end = 0

        while user_batch_start < len(self.usersToEvaluate):

            user_batch_end = user_batch_start + block_size
            user_batch_end = min(user_batch_end, len(self.usersToEvaluate))

            test_user_batch_array = np.array(self.usersToEvaluate[user_batch_start:user_batch_end])
            user_batch_start = user_batch_end

            # Compute predictions for a batch of users using vectorization, much more efficient than computing it one at a time
            recommended_items_batch_list, all_items_predicted_ratings = recommender_object.recommend(test_user_batch_array,
                                                                        remove_seen_flag=self.exclude_seen,
                                                                        cutoff=self.max_cutoff,
                                                                        remove_top_pop_flag=False,
                                                                        remove_custom_items_flag=self.ignore_items_flag,
                                                                        return_scores=True)

            # Compute recommendation quality for each user in batch
            for batch_user_index in range(len(recommended_items_batch_list)):

                user_id = test_user_batch_array[batch_user_index]
                recommended_items = np.array(recommended_items_batch_list[batch_user_index])
                predicted_ratings = all_items_predicted_ratings[batch_user_index].flatten()

                # Being the URM CSR, the indices are the non-zero column indexes
                relevant_items = self.get_user_relevant_items(user_id)
                relevant_items_ratings = self.get_user_test_ratings(user_id)

                n_users_evaluated += 1

                metrics_handler.add_user_evaluation(user_id, recommended_items, predicted_ratings,
                                                    relevant_items, relevant_items_ratings)

                if time.time() - start_time_print > 30 or n_users_evaluated == len(self.usersToEvaluate):
                    elapsed_time = time.time()-start_time
                    new_time_value, new_time_unit = seconds_to_biggest_unit(elapsed_time)
                    print(
                        "{}: Processed {} ( {:.2f}% ) in {:.2f} {}. Users per second: {:.0f}".format(
                        self.EVALUATOR_NAME, n_users_evaluated,
                        100.0* float(n_users_evaluated)/len(self.usersToEvaluate),
                        new_time_value, new_time_unit,
                        float(n_users_evaluated)/(time.time()-start_time))
                    )

                    sys.stdout.flush()
                    sys.stderr.flush()

                    start_time_print = time.time()

        return metrics_handler


    def evaluateRecommender(self, recommender_object, URM_test=None, ignore_users=None, ignore_items=None):
        """
        :param recommender_object: the trained recommender object, a Recommender subclass
        :param URM_test: list of URMs to test the recommender against, or a single URM object
        :param cutoff_list: list of cutoffs to be use to report the scores, or a single cutoff
        """

        super(EvaluatorHoldout, self).evaluateRecommender(recommender_object, URM_test, ignore_users, ignore_items)

        if self.ignore_items_flag:
            recommender_object.set_items_to_ignore(self.ignore_items_ID)

        metrics_handler = self._run_evaluation_on_selected_users(recommender_object)

        if metrics_handler.get_evaluated_users_count() <= 0:
            print("{} WARNING: No users had a sufficient number of relevant items".format(self.EVALUATOR_NAME))

        if self.ignore_items_flag:
            recommender_object.reset_items_to_ignore()

        return metrics_handler

