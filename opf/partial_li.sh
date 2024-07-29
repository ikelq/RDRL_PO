#!/bin/bash
#SBATCH --export=ALL
#SBATCH --job-name=er3pf
#SBATCH -o out3pf -e err3pf
#SBATCH -N 1 -n 3 -t 240:00:00 --mem 100mb -c 1
#SBATCH --mail-user=liuqiong_yl@outllok.com
#SBATCH -p allcpu

# python MBO15.py --env_name 33 & 
# python MBO15.py --env_name 69 & 
# python MBO15.py --env_name 118 & 

python MBO15_without_control.py --env_name 33 & 
python MBO15_without_control.py --env_name 69 & 
python MBO15_without_control.py --env_name 118 & 

wait
