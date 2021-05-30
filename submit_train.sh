#!/bin/bash
#SBATCH --job-name=7_gru
#SBATCH --output=7_gru.log
#SBATCH --error=7_gru.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --partition=skylake-gpu
# require 2 GPU
#SBATCH --gres=gpu:2
#SBATCH --time=30:00:00

module load anaconda3/5.1.0
source activate gpu_testing
module load cudnn/8.1.0-cuda-11.2.0
module load cuda/11.2.0

cd "/fred/oz138/COS80028/P1/ConvLSTM1D"

srun "/home/hlai/.conda/envs/gpu_testing/bin/python" "Ozstar_training_72000_bi_gru_v1.py"