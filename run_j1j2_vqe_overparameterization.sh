#!/bin/bash
conda activate qlt
python3 run_vqe_overparameterization.py 3 4
python3 run_vqe_overparameterization.py 3 5
python3 run_vqe_overparameterization.py 3 6
python3 run_vqe_overparameterization.py 5 4
python3 run_vqe_overparameterization.py 5 5
python3 run_vqe_overparameterization.py 5 6
python3 run_vqe_overparameterization.py 6 4
python3 run_vqe_overparameterization.py 6 5
python3 run_vqe_overparameterization.py 6 6
