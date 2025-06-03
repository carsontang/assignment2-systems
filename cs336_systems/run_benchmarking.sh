#! /bin/bash

uv run python ~/Documents/Github/assignment2-systems/cs336_systems/benchmarking_script.py \
    --batchsize 3 \
    --vocab 4 \
    --context 5 \
    --dmodel 768 \
    --nlayers 12 \
    --nheads 12 \
    --dff 3072 \
	--use-random-seed \
    --num-runs 10 \
    --num-warmup 2