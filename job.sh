#!/bin/sh
#PBS -q F1accs
#PBS -l select=1:ncpus=64
module purge
module load cuda/11.2
source myenv/bin/activate
cd examples
time python3 mnist.py

