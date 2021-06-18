#!/bin/sh
#BSUB -q gpua100
#BSUB -J "Cycle_GAN"
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 23:59
#BSUB -R "rusage[mem=8GB]"
##BSUB -B
### -- send notification at completion--
##BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -oo cycleGAN.out
##BSUB -eo gpu-%J.err
# -- end of LSF options --

nvidia-smi
# Load the cuda module
module load cuda/11.1
module load cudnn/v8.0.4.30-prod-cuda-11.1

source venv/bin/activate
python cycle_GAN_rasmus.py
