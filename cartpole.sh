#!/bin/sh
# hidden = 20
python3 ./main.py \
    --w 4 \
    --v 1 \
    --f 2 \
    --gpu 0 \
    --out logs/cartpole \
    --seed 0 \
    --env CartPole-v1 \
    --el 1000 \
    --rl 50 \
    --rb -1 \
    --ent 1e-10 \
    --dr 1 \
    --td 0 \
    --lr 1e-3 \
    --save-model 1 \
    --save-plot 1