#!/bin/bash
#SBATCH -J VNL_Train
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --output=log.%j.out
#SBATCH --error=log.%j.err
#SBATCH -t 30:00:00
#SBATCH --gres=gpu:2
module load anaconda3/2019.07
module unload cuda/10.0
module load cuda/9.0
source activate xiaor
python -u train.py --tub ../data/data_11_20_1521/tub/ --model ./models/mypilot2
# python -u modelconvert.py
