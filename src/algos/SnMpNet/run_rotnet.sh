#!/bin/bash
#SBATCH -p gpu20
#SBATCH --gres gpu:1
#SBATCH -o /BS/UCDR/work/slurm/rotnet/slurm-%A_%a.out
#SBATCH -t 2-0

cmd="python3 ttt_rotnet.py -sd quickdraw -hd sketch -gd real -e 1 -bs 64 -lrc 5e-5 -lrb 5e-6"
# cmd="python3 ttt_rotnet.py -sd quickdraw -hd sketch -gd real -e 1 -bs 64 -lrc 1e-5 -lrb 0"
# cmd="python3 ttt_rotnet.py -sd quickdraw -hd sketch -gd real -e 1 -bs 64 -lrc 5e-5 -lrb 0"
# cmd="python3 ttt_rotnet.py -sd quickdraw -hd sketch -gd real -e 1 -bs 64 -lrc 1e-5 -lrb 1e-6"

# cmd="python3 ttt_rotnet.py -sd sketch -hd quickdraw -gd real -e 1 -bs 64 -lrc 5e-5 -lrb 5e-6"
# cmd="python3 ttt_rotnet.py -sd painting -hd clipart -gd real -e 1 -bs 64 -lrc 5e-5 -lrb 5e-6"
# cmd="python3 ttt_rotnet.py -sd painting -hd infograph -gd real -e 1 -bs 64 -lrc 5e-5 -lrb 5e-6"
# cmd="python3 ttt_rotnet.py -sd infograph -hd painting -gd real -e 1 -bs 64 -lrc 5e-5 -lrb 5e-6"

# print start time and command to log
echo $(date)
echo $cmd


# start command
$cmd