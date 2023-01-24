#!/bin/bash
#SBATCH -p gpu20
#SBATCH --gres gpu:1
#SBATCH -o /BS/UCDR/work/slurm/jig2/cross/slurm-%A_%a.out
#SBATCH -t 2-0

# cmd="python3 jig_sketchy_on_domainnet.py -sd quickdraw -hd sketch -gd real -e 1 -bs 64 -lrb 5e-5"
# cmd="python3 jig_sketchy_on_domainnet.py -sd sketch -hd quickdraw -gd real -e 1 -bs 64 -lrb 5e-5"
# cmd="python3 jig_sketchy_on_domainnet.py -sd painting -hd clipart -gd real -e 1 -bs 64 -lrb 5e-5"
# cmd="python3 jig_sketchy_on_domainnet.py -sd painting -hd infograph -gd real -e 1 -bs 64 -lrb 5e-5"
cmd="python3 jig_sketchy_on_domainnet.py -sd infograph -hd painting -gd real -e 1 -bs 64 -lrb 5e-5"

# print start time and command to log
echo $(date)
echo $cmd


# start command
$cmd