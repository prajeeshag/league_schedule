#!/bin/bash
#PBS -N vfl
#PBS -q cccr
#PBS -l walltime=20:00:00
#PBS -W depend=afterok:
#PBS -l place=scatter
#PBS -l select=1:ncpus=36
cd /home/cccr/prajeesh/league_schedule
rm -f *.o* *.e*
source .venv/bin/activate

aprun python schedule.py -t 14 -d 26 --matches_per_day 7 --cpu 72 > output.log