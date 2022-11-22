#!/bin/bash
#SBATCH -J j1j2-vqe-cma-es-hyperparameterization
#SBATCH -p batch
#SBATCH --time=7-00:00:00
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=10000
module load anaconda/3
source activate /cluster/tufts/lovelab/wsimon02/condaenv/qlt
python3 cma-es-hyperparameter-optimization.py 5 5 16 & 
python3 cma-es-hyperparameter-optimization.py 5 5 32 & 
python3 cma-es-hyperparameter-optimization.py 5 5 64 & 
python3 cma-es-hyperparameter-optimization.py 5 5 128 & 
python3 cma-es-hyperparameter-optimization.py 5 5 256 & 
python3 cma-es-hyperparameter-optimization.py 5 5 512 & 
python3 cma-es-hyperparameter-optimization.py 5 5 1024 

