#!/bin/sh
# el = 2000
python3 ../main.py \
    --w 4 \
    --v 1 \
    --f 60 \
    --gpu 0 \
    --out logs/mountaincar \
    --seed 0 \
    --env MountainCar-v0 \
    --el 1000 \
    --rl 3500 \
    --rb -1 \
    --ent 5e-1 \
    --dr 0.99 \
    --td 0 \
    --lr 1e-4 \
    --save-model 1 \
    --save-plot 1