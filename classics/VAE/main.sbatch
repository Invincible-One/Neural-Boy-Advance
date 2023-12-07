#!/bin/bash

#SBATCH --job-name=VAE
#SBATCH --output=VAE.out

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=24GB
#SBATCH --gres=gpu:rtx8000:1

#SBATCH --mail-type=ALL
#SBATCH --mail-user=ym2380@nyu.edu

module purge
singularity exec --nv --overlay /scratch/ym2380/my_env/pytorch_env.ext3:ro /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash \
-c "source /ext3/env.sh; python main.py"
