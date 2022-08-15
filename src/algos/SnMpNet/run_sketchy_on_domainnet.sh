#!/bin/bash
#SBATCH -p gpu20
#SBATCH --gres gpu:1
#SBATCH -o /BS/UCDR/work/slurm/cross/slurm-%A_%a.out
#SBATCH -t 2-0

cmd="python3 sketchy_on_domainnet.py -sd quickdraw -hd sketch -gd real"
# cmd="python3 sketchy_on_domainnet.py -sd sketch -hd quickdraw -gd real"
# cmd="python3 sketchy_on_domainnet.py -sd painting -hd infograph -gd real"
# cmd="python3 sketchy_on_domainnet.py -sd infograph -hd painting -gd real"

# print start time and command to log
echo $(date)
echo $cmd


# start command
$cmd