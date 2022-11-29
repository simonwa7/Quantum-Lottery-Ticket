#!/bin/bash
#SBATCH -J qlt-larocca-overparameterization
#SBATCH -p batch
#SBATCH --time=7-00:00:00
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=10000
module load anaconda/3
source activate /cluster/tufts/lovelab/wsimon02/condaenv/qlt
python3 ../run_larocca_overparameterization_input.py 5 20
python3 ../run_larocca_overparameterization_input.py 5 100