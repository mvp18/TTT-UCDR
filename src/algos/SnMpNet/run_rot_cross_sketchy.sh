#!/bin/bash
#SBATCH -p gpu20
#SBATCH --gres gpu:1
#SBATCH -o /BS/UCDR/work/slurm/rotnet/slurm-%A_%a.out
#SBATCH -t 2-0

# cmd="python3 rot_domainnet_on_sketchy.py -e 1 -bs 64 -lrc 1e-5 -lrb 0 -eccv 1 -rot_v 2"
# cmd="python3 rot_domainnet_on_sketchy.py -e 1 -bs 64 -lrc 1e-5 -lrb 0 -eccv 1 -rot_v 1"
# cmd="python3 rot_domainnet_on_sketchy.py -e 1 -bs 64 -lrc 5e-6 -lrb 0 -eccv 1 -rot_v 2"
# cmd="python3 rot_domainnet_on_sketchy.py -e 1 -bs 64 -lrc 5e-6 -lrb 0 -eccv 1 -rot_v 1"

# cmd="python3 rot_domainnet_on_sketchy.py -e 1 -bs 64 -lrc 1e-5 -lrb 1e-6 -eccv 1 -rot_v 2"
cmd="python3 rot_domainnet_on_sketchy.py -e 1 -bs 64 -lrc 1e-5 -lrb 1e-6 -eccv 1 -rot_v 1"

# print start time and command to log
echo $(date)
echo $cmd


# start command
$cmd