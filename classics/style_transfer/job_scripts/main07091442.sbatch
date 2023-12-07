#!/bin/bash

#SBATCH --job-name=style1442
#SBATCH --output=../logs/07091442.out

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --mem=24GB
#SBATCH --gres=gpu:rtx8000:1

#SBATCH --mail-type=ALL
#SBATCH --mail-user=ym2380@nyu.edu

module purge
singularity exec --nv --overlay /scratch/ym2380/my_env/pytorch_env.ext3:ro /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash \
-c "source /ext3/env.sh; python ../code/main.py --content "content002.JPG" --style "style001.jpeg" --working_dir "content001_style001""
