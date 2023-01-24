#!/bin/bash
#SBATCH -p gpu20
#SBATCH --gres gpu:1
#SBATCH -o /BS/UCDR/work/slurm/rotnet/slurm-%A_%a.out
#SBATCH -t 2-0

# cmd="python3 rot_sketchy_on_domainnet.py -hd quickdraw -sd sketch -gd real -e 1 -bs 64 -lrc 1e-5 -lrb 1e-6 -rot_v 2"
# cmd="python3 rot_sketchy_on_domainnet.py -hd quickdraw -sd sketch -gd real -e 1 -bs 64 -lrc 1e-5 -lrb 1e-6 -rot_v 1"

# cmd="python3 rot_sketchy_on_domainnet.py -hd painting -sd infograph -gd real -e 1 -bs 64 -lrc 1e-5 -lrb 1e-6 -rot_v 1"
# cmd="python3 rot_sketchy_on_domainnet.py -hd painting -sd infograph -gd real -e 1 -bs 64 -lrc 1e-5 -lrb 1e-6 -rot_v 2"

# cmd="python3 rot_sketchy_on_domainnet.py -hd infograph -sd painting -gd real -e 1 -bs 64 -lrc 1e-5 -lrb 1e-6 -rot_v 1"
# cmd="python3 rot_sketchy_on_domainnet.py -hd infograph -sd painting -gd real -e 1 -bs 64 -lrc 1e-5 -lrb 1e-6 -rot_v 2"

# cmd="python3 rot_sketchy_on_domainnet.py -hd clipart -sd painting -gd real -e 1 -bs 64 -lrc 1e-5 -lrb 1e-6 -rot_v 1"
cmd="python3 rot_sketchy_on_domainnet.py -hd clipart -sd painting -gd real -e 1 -bs 64 -lrc 1e-5 -lrb 1e-6 -rot_v 2"

# print start time and command to log
echo $(date)
echo $cmd

# start command
$cmd