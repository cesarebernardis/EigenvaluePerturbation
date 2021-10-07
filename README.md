# Eigenvalue Perturbation

This project was developed by [Cesare Bernardis](https://scholar.google.it/citations?user=9fzJj_AAAAAJ), 
Ph.D. candidate at Politecnico di Milano, and it is based on the recommendation framework used by our [research group](http://recsys.deib.polimi.it/).
The code allows reproducing the results of "Eigenvalue Perturbation for Item-based Recommender Systems", 
a work presented at the ACM RecSys Conference 2021.

Please cite our article ([BibTex](https://dblp.org/rec/conf/recsys/BernardisC21.html?view=bibtex), [ACM DL](https://dl.acm.org/doi/10.1145/3460231.3478862)) if you use this repository or a part of it in your work.


## Installation

---

To run the experiments, we suggest creating an ad hoc virtual environment using [Anaconda](https://www.anaconda.com/).
 
_You can also use other types of virtual environments (e.g. virtualenv). 
Just ensure to install the packages listed in 'requirements.txt' in the environment used to run the experiments._

To install Anaconda you can follow the instructions on the [website](https://www.anaconda.com/products/individual).


To create the Anaconda environment with all the required python packages, run the following command:

```console
bash create_env.sh
```

The default name for the environment is _recsys-eigenpert_.
You can change it by replacing the string _recsys-eigenpert_ in 'create_env.sh' with the name you prefer.
All the following commands have to be modified according to the new name.

Note that this framework includes some [Cython](https://cython.readthedocs.io/en/latest/index.html) implementations that need to be compiled.
Before continuing with next steps, ensure to have a C/C++ compiler installed.
On several Linux distributions (e.g. Ubuntu, Debian) it is enough to run the following:

```console
sudo apt-get install build-essential
```

Finally, to install the framework you can simply activate the created virtual environment and execute the 'install.sh' script, using the following commands:

```console
conda activate recsys-eigenpert
sh install.sh
```

## Run the experiments

---

First of all, you should activate the environment where you installed the framework.
If you followed the installation guide above, the command is:

```console
conda activate recsys-eigenpert
```

Then, the first step to run the experiments consists in downloading the data and generating the splits.
These actions are performed by the 'create_splits.py' script that can be executed with the following command:

```console
python create_splits.py
```


To perform the evaluation on the splits created, run:

```console
python scripts/run_test.py
```

The results are saved in a CSV file called 'optimized_results.csv'.

Automatically, the script employs the best hyper-parameter configurations found by the optimization procedure we performed in our experiments.
You can find these configurations in 'data/best-hyperparameters.zip'.

If you want to perform a new hyper-parameter optimization, you can use:

```console
python scripts/run_parameter_search.py
```

