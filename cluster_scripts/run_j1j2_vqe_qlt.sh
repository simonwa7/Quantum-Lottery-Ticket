#!/bin/bash
#SBATCH -J j1j2-vqe-qlt
#SBATCH -p batch
#SBATCH --time=7-00:00:00
#SBATCH -n 1
#SBATCH -c 3
#SBATCH --mem=10000
module load anaconda/3
source activate /cluster/tufts/lovelab/wsimon02/condaenv/qlt
python3 run_j1j2_vqe_qlt.py 5 5 "CMA-ES"

