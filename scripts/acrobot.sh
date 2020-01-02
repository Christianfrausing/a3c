#!/bin/sh
# hidden = 20
python3 ../main.py \
    --w 4 \
    --v 1 \
    --f 5 \
    --gpu 0 \
    --out logs/acrobot \
    --seed 0 \
    --env Acrobot-v1 \
    --el 500 \
    --rl 500 \
    --rb -1 \
    --ent 0.1 \
    --dr 1 \
    --td 0 \
    --lr 1.5e-3 \
    --save-model 1 \
    --save-plot 1