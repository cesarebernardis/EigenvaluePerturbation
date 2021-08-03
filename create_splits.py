import numpy as np
import os

from RecSysFramework.Utils import invert_dictionary

from RecSysFramework.ExperimentalConfig import EXPERIMENTAL_CONFIG


for splitter in EXPERIMENTAL_CONFIG['splits']:
    for dataset_config in EXPERIMENTAL_CONFIG['datasets']:

        datareader = dataset_config['datareader']()
        postprocessings = dataset_config['postprocessings']

        dataset = datareader.load_data(postprocessings=postprocessings)
        dataset.save_data()

        np.random.seed(42)
        train, test, validation = splitter.split(dataset)
        splitter.save_split([train, test, validation])

        for fold in range(EXPERIMENTAL_CONFIG['n_folds']):

            np.random.seed(fold+1)
            train, test, validation = splitter.split(dataset)
            splitter.save_split([train, test, validation], filename_suffix="_{}".format(fold))
