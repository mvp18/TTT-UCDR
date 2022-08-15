#!/bin/bash
#SBATCH -p gpu20
#SBATCH --gres gpu:1
#SBATCH -o /BS/UCDR/work/Rotnet_results/slurm-%A_%a.out
#SBATCH -t 2-0

#cmd="python3 ttt_rotnet.py -sd quickdraw -hd sketch -gd real -e 5 -bs 64 -lrc 5e-6"
#cmd="python3 ttt_rotnet.py -sd sketch -hd quickdraw -gd real -e 5 -bs 64 -lrc 5e-6"
#cmd="python3 ttt_rotnet.py -sd painting -hd clipart -gd real -e 5 -bs 64 -lrc 5e-6"
#cmd="python3 ttt_rotnet.py -sd painting -hd infograph -gd real -e 5 -bs 64 -lrc 5e-6"
#cmd="python3 ttt_rotnet.py -sd infograph -hd painting -gd real -e 5 -bs 64 -lrc 5e-6"
#cmd="python3 main.py -hd sketch -sd quickdraw -gd real -aux 1 -wcce 1 -wmse 1 -alpha 1 -wrat 1 -wrot 1 -beta 1 -bs 30 -mixl img -es 15"
cmd="python3 main.py -hd quickdraw -sd sketch -gd real -aux 1 -wcce 1 -wmse 1 -alpha 1 -wrat 1 -wrot 1 -beta 2 -bs 30 -mixl img -es 15"
#cmd="python3 main.py -hd painting -sd infograph -gd real -aux 1 -wcce 1 -wmse 1 -alpha 1 -wrat 1 -wrot 1 -beta 1 -bs 30 -mixl img -es 15"
#cmd="python3 main.py -hd infograph -sd painting -gd real -aux 1 -wcce 1 -wmse 1 -alpha 1 -wrat 1 -wrot 1 -beta 1 -bs 30 -mixl img -es 15"
#cmd="python3 main.py -hd clipart -sd painting -gd real -aux 1 -wcce 1 -wmse 1 -alpha 1 -wrat 1 -wrot 1 -beta 1 -bs 30 -mixl img -es 15"

# print start time and command to log
echo $(date)
echo $cmd


# start command
$cmd