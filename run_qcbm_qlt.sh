#!/bin/bash
#SBATCH -J qcbm-qlt
#SBATCH -p batch
#SBATCH --time=7-00:00:00
#SBATCH -n 1
#SBATCH -c 3
#SBATCH --mem=10000
module load anaconda/3
source activate /cluster/tufts/lovelab/wsimon02/condaenv/qlt
python3 run_qcbm_qlt.py 5 4

