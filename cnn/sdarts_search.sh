#!/bin/bash
#SBATCH -J darts_search
#SBATCH -p gpu
#SBATCH -n 8
#SBATCH --gres=gpu:1 # number of gpu
#SBATCH --gpu-freq=high
#SBATCH --mem 10000 # Memory request (10Gb)
#SBATCH -t 0-24:00 # Maximum execution time (D-HH:MM)
#SBATCH -o darts_search.out # Standard output
#SBATCH -e darts_search.err # Standard error
module load Anaconda3/5.0.1-fasrc02
module load cuda/10.0.130-fasrc01 cudnn/7.4.1.5_cuda10.0-fasrc01
module load gcc/8.2.0-fasrc01
source activate darts
python train_search.py --dataset mnist --save MNIST
