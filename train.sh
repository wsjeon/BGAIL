#!/bin/bash
source /home/wsjeon/.bashrc
source /home/wsjeon/.profile
cd $2
/home/wsjeon/anaconda3/envs/bgail/bin/python train_multiple_runs.py --process_id $1