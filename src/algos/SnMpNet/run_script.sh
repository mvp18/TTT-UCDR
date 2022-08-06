#!/bin/bash
#SBATCH -p gpu20
#SBATCH --gres gpu:1
#SBATCH -o /BS/UCDR/work/slurm/slurm-%A_%a.out
#SBATCH -t 2-0

cmd="python3 test.py -sd quickdraw -hd sketch -gd real"

# print start time and command to log
echo $(date)
echo $cmd


# start command
$cmd