#!/bin/bash
# tmux session name
name=bgail

# Kill tmux session with $name, if it exists.
tmux kill-session -t $name

# Make new tmux sessions with $name.
tmux new-session -s $name -n w0 -d bash

# Make new windows
tmux new-window -t $name -n main -d bash # main
tmux new-window -t $name -n tb -d bash # tensorboard
tmux new-window -t $name -n chrome -d bash # chrome

# Empty window for killing tmux
tmux send-keys -t $name:w0\
        "tmux kill-session" # No Enter here

# Main window for condor jobs
tmux send-keys -t $name:main\
        "ssh wsjeon@havok.kaist.ac.kr"\
        Enter
tmux send-keys -t $name:main\
        "cd $PWD"\
        Enter
tmux send-keys -t $name:main\
        "condor_submit TMPPWD=$PWD submit"\
        Enter

## ssh tunneling port for tensorboard
#PORT=5500 # ssh tunneling port
#tmux send-keys -t $name:tb\
#       "ssh -L $PORT:127.0.0.1:$PORT wsjeon@havok.kaist.ac.kr"\
#               Enter
#tmux send-keys -t $name:tb\
#       "ssh -L $PORT:127.0.0.1:$PORT wsjeon@c15"\
#               Enter
#tmux send-keys -t $name:tb\
#       "tensorboard --logdir=/home/wsjeon/code/agent-log/main-gail-tf-mpi --port $PORT"\
#               Enter
#
## waiting for tensorboard initialization
#pause 15
#
## chrome to see tensorboard in local
#tmux send-keys -t $name:chrome\
#       "google-chrome 127.0.0.1:$PORT"\
#               Enter
tmux a
