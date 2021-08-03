conda config --add channels conda-forge

conda create -y -n recsys-eigenpert --file requirements.txt python=3.6

source activate recsys-eigenpert

pip install similaripy

cd ../../../..
