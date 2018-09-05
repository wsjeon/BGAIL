#!/bin/bash
source /home/wsjeon/.bashrc
source /home/wsjeon/.profile
cd $2
PYTHONDONTWRITEBYTECODE=1 /home/wsjeon/anaconda3/envs/bgail/bin/python -B train_multiple_runs.py --process_id $1