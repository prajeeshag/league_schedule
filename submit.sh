#!/bin/bash
#PBS -N vfl
#PBS -q cccr
#PBS -l walltime=20:00:00
#PBS -W depend=afterok:
#PBS -l place=scatter
#PBS -l select=1:ncpus=36
cd /home/cccr/prajeesh/constrain_programming
rm -f *.o* *.e*
source .venv/bin/activate

aprun python schdl.py --cpu 72  > output.log 2>&1