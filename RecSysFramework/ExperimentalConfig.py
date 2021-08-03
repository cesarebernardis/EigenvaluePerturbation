from RecSysFramework.DataManager.Reader import BookCrossingReader
from RecSysFramework.DataManager.Reader import Movielens20MReader
from RecSysFramework.DataManager.Reader import LastFMHetrec2011Reader
from RecSysFramework.DataManager.Reader import EpinionsReader

from RecSysFramework.DataManager.Splitter import LeaveKOut, Holdout
from RecSysFramework.DataManager.DatasetPostprocessing import KCore, ImplicitURM

from RecSysFramework.Evaluation import EvaluatorMetrics

from RecSysFramework.Recommender.KNN import ItemKNNCF, EASE_R
from RecSysFramework.Recommender.GraphBased import P3alpha, RP3beta
from RecSysFramework.Recommender.SLIM.ElasticNet import SLIM


EXPERIMENTAL_CONFIG = {
    'n_folds': 10,
    'splits': [
        Holdout(train_perc=0.6, validation_perc=0.2, test_perc=0.2),
    ],
    'datasets': [
        {
            'datareader': Movielens20MReader,
            'postprocessings': [
                ImplicitURM(min_rating_threshold=3.),
                KCore(user_k_core=5, item_k_core=5, reshape=True),
            ]
        }, {
            'datareader': BookCrossingReader,
            'postprocessings': [
                ImplicitURM(min_rating_threshold=6.5),
                KCore(user_k_core=5, item_k_core=5, reshape=True),
            ]
        }, {
            'datareader': LastFMHetrec2011Reader,
            'postprocessings': [
                ImplicitURM(min_rating_threshold=0.),
                KCore(user_k_core=5, item_k_core=5, reshape=True),
            ]
        }, {
            'datareader': EpinionsReader,
            'postprocessings': [
                ImplicitURM(min_rating_threshold=3.),
                KCore(user_k_core=5, item_k_core=5, reshape=True),
            ]
        }

    ],
    'item-based-algorithms': [ItemKNNCF, P3alpha, RP3beta, SLIM, EASE_R],
    'recap_metrics': [
        EvaluatorMetrics.PRECISION,
        EvaluatorMetrics.RECALL,
        EvaluatorMetrics.MAP,
        EvaluatorMetrics.NDCG,
        EvaluatorMetrics.COVERAGE_ITEM_TEST
    ],
    'cutoffs': [5, 10, 25],
}
