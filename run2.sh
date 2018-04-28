#!/bin/sh
export CUDA_VISIBLE_DEVICES=0
# python -u main.py --model QRNN --batch_size 20 --clip 0.2 --wdrop 0.1 --nhid 1550 --nlayers 4 --emsize 400 --dropouth 0.2 --seed 9001 --dropouti 0.4 --epochs 550 --save PTB_em_also_dropout.pt --log-interval 100 --dropout 0.3 --nonmono 10
# python -u main.py --model QRNN --batch_size 20 --clip 0.2 --wdrop 0.1 --nhid 1550 --nlayers 4 --emsize 400 --dropouth 0.2 --seed 9001 --dropouti 0.4 --epochs 550 --save PTB1.pt --resume PTB1.pt --log-interval 100 --dropout 0.3 --nonmono 10 --test
#1 python -u main.py --model QRNN --batch_size 20 --clip 0.2 --wdrop 0.1 --nhid 1550 --nlayers 4 --emsize 400 --dropouth 0.3 --seed 9001 --dropouti 0.4 --epochs 950 --save PTB1.pt --log-interval 100 --nonmono 10
# python -u main.py --model QRNN --batch_size 20 --clip 0.2 --wdrop 0.1 --nhid 1000 --nlayers 4 --emsize 400 --dropouth 0.3 --seed 9001 --dropouti 0.4 --epochs 450 --save PTB_1000.pt --resume PTB_1000.pt --test --log-interval 100 --no_md # Train from scratch
# python -u main.py --model QRNN --batch_size 20 --clip 0.2 --wdrop 0.1 --nhid 1200 --nlayers 4 --emsize 400 --dropouth 0.3 --seed 9001 --dropouti 0.4 --epochs 550 --save PTB_1200.pt --log-interval 100 --no_md # From scratch
# python -u main.py --model QRNN --batch_size 20 --clip 0.2 --wdrop 0.076 --nhid 1550 --nlayers 4 --emsize 400 --dropouth 0.228 --seed 9001 --dropouti 0.4 --epochs 950 --save PTB2.pt --log-interval 100 --nonmono 10 --dropout 0.304 --resume PTB2.pt --test # current best (59.69)

# python -u main.py --model QRNN --batch_size 20 --clip 0.2 --wdrop 0.1 --nhid 1550 --nlayers 4 --emsize 400 --dropouth 0.3 --seed 9001 --dropouti 0.4 --epochs 950 --save PTB2.pt --log-interval 100 --nonmono 5 --resume PTB2.pt --keep_pct 0.4 --no_md --log_out PTB_retrain_qrnn --refresh_opt --optimizer asgd # current best (59.69) prune then retrain 40%
# python -u main.py --model QRNN --batch_size 20 --clip 0.2 --wdrop 0.1 --nhid 1550 --nlayers 4 --emsize 400 --dropouth 0.3 --seed 9001 --dropouti 0.4 --epochs 950 --save PTB2.pt --log-interval 100 --nonmono 5 --resume PTB2.pt --keep_pct 0.6 --no_md --log_out PTB_retrain_qrnn --refresh_opt --optimizer asgd # current best (59.69) prune then retrain 60%


# WIKITEXT-2
# python -u main.py --epochs 600 --data data/wikitext-2 --clip 0.25 --dropouti 0.4 --dropouth 0.16 --nhid 1550 --nlayers 4 --seed 4002 --model QRNN --wdrop 0.076 --batch_size 20 --save WT2.pt --nonmono 10 --dropout 0.304 --log-interval 100 --log_out wt2_md_qrnn_2
# python -u main.py --epochs 600 --data data/wikitext-2 --clip 0.25 --dropouti 0.4 --dropouth 0.2 --nhid 1550 --nlayers 4 --seed 4002 --model QRNN --wdrop 0.1 --batch_size 40 --save WT2_3.pt --resume WT2_3.pt --optimizer asgd --refresh_opt --nonmono 10 --log-interval 100 --log_out wt2_md_qrnn_3 --test --keep_pct_test 0.2 # BEST WT-2

# python -u main.py --epochs 550 --data data/wikitext-2 --clip 0.25 --dropouti 0.4 --dropouth 0.2 --nhid 1550 --nlayers 4 --seed 4002 --model QRNN --wdrop 0.1 --batch_size 40 --save WT2_4.pt --nonmono 10 --log_out wt2_qrnn --no_md
python -u main.py --epochs 500 --data data/wikitext-2 --clip 0.25 --dropouti 0.4 --dropouth 0.2 --nhid 1550 --nlayers 4 --seed 4002 --model QRNN --wdrop 0.1 --batch_size 40 --save WT2_4.pt --resume WT2_4.pt --nonmono 5 --optimizer asgd --refresh_opt --log_out wt2_qrnn --no_md



# python -u main.py --model QRNN --batch_size 20 --clip 0.2 --wdrop 0.1 --nhid 1550 --nlayers 4 --emsize 400 --dropouth 0.15 --seed 9001 --dropouti 0.4 --epochs 550 --save PTB2.pt --resume PTB2.pt --log-interval 100 --dropout 0.2 --nonmono 10 --optimizer asgd
# python -u main.py --model QRNN --batch_size 20 --clip 0.2 --wdrop 0.1 --nhid 1550 --nlayers 4 --emsize 400 --dropouth 0.3 --seed 9001 --dropouti 0.4 --epochs 550 --save PTB.pt --resume PTB3.pt --log-interval 100 --nonmono 10

# QRNN-MD WT103
# python -u main.py --epochs 14 --nlayers 4 --emsize 400 --nhid 2500 --alpha 0 --beta 0 --dropoute 0 --dropouth 0.1 --dropouti 0.1 --dropout 0.1 --wdrop 0 --wdecay 0 --bptt 140 --batch_size 50 --optimizer adam --lr 1e-3 --data data/wikitext-103 --save WT103.12hr.QRNN.pt --when 12 --model QRNN --split_cross

# LSTM-MD
# python main.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.175 --wdrop 0.076 --seed 141 --epoch 500 --model LSTM-MD --log-interval 100 --dropout 0.304 --save LSTM_PTB_rescaled.pt
# python main.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.2 --seed 141 --epoch 650 --model LSTM-MD --log-interval 100 --save LSTM_PTB_default.pt --nonmono 10 --wdrop 0.08 --dropout 0.35 --log_out results_md_awd
# python main.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --epoch 650 --model LSTM-MD --log-interval 100 --save LSTM_PTB_default.pt --resume LSTM_PTB_default.pt --nonmono 10 --test #--log_out results_md_awd_regular
# python main.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --epoch 650 --model LSTM-MD --log-interval 100 --save LSTM_PTB_default_rescale.pt --nonmono 10 --test --resume LSTM_PTB_default_rescale.pt #--log_out results_md_awd_rescale

# python main.py --batch_size 20 --data data/penn --dropouti 0.4 --dropouth 0.25 --seed 141 --epoch 650 --model LSTM --log-interval 100 --save LSTM_PTB_default.pt --nonmono 10
