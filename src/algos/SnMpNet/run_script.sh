#!/bin/bash
#SBATCH -p gpu20
#SBATCH --gres gpu:1
#SBATCH -o /BS/UCDR/work/slurm/slurm-%A_%a.out
#SBATCH -t 2-0

# cmd="python3 ttt_barlow.py -sd quickdraw -hd sketch -gd real -e 3 -bs 128 -lrc 1e-5"
# cmd="python3 ttt_barlow.py -sd sketch -hd quickdraw -gd real -e 3 -bs 128 -lrc 1e-5"
# cmd="python3 ttt_barlow.py -sd painting -hd clipart -gd real -e 3 -bs 128 -lrc 1e-5"
# cmd="python3 ttt_barlow.py -sd painting -hd infograph -gd real -e 3 -bs 128 -lrc 1e-5"
cmd="python3 ttt_barlow.py -sd infograph -hd painting -gd real -e 3 -bs 128 -lrc 1e-5"

# print start time and command to log
echo $(date)
echo $cmd


# start command
$cmd