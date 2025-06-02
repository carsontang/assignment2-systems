#! /bin/bash

uv run python ~/Documents/Github/assignment2-systems/cs336_systems/benchmarking_script.py \
    --vocab 10000 \
    --context 32 \
    --dmodel 768 \
    --nlayers 12 \
    --nheads 12 \
    --dff 3072 \
		--use-random-seed
