#!/bin/sh
CUDA_VISIBLE_DEVICES=0 python -u main.py --model QRNN --batch_size 20 --clip 0.2 --wdrop 0.1 --nhid 1550 --nlayers 4 --emsize 400 --dropouth 0.3 --seed 9001 --dropouti 0.4 --epochs 550 --scale_alpha 1 --binarized --save PTB.pt --log-interval 50