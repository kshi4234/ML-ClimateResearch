#!/bin/bash

#SBATCH --account=csci_ga_2565-2024sp
#SBATCH --partition=n1s8-v100-1
#SBATCH --gres=gpu:v:100:1
#SBATCH --time=1:00:00
#SBATCH --job-name=pretrain

module purge

singularity exec --nv \
	    --overlay /scratch/<NetID>/pytorch-example/my_pytorch.ext3:ro \
	    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
	    /bin/bash -c "source /ext3/env.sh; python src/climax/pretrain/train.py --config configs/pretrain_climax.yaml --trainer.strategy=ddp --trainer.devices=1 --trainer.max_epochs=100 --data.batch_size=16 --model.lr=5e-4 --model.beta_1="0.9" --model.beta_2="0.95" --model.weight_decay=1e-5"
