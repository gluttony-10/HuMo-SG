#!/bin/bash

# torchrun --node_rank=0 --nproc_per_node=8 --nnodes=1 \
CUDA_VISIBLE_DEVICES=0 torchrun --node_rank=0 --nproc_per_node=1 --nnodes=1 \
    --rdzv_endpoint=127.0.0.1:12345 \
    --rdzv_conf=timeout=900,join_timeout=900,read_timeout=900 \
    main.py humo/configs/inference/generate_1_7B.yaml \
    generation.frames=97 \
    generation.scale_t=7.5 \
    generation.scale_i=4.0 \
    generation.scale_a=7.5 \
    generation.mode=TIA \
    generation.height=480 \
    generation.width=832 \
    diffusion.timesteps.sampling.steps=50 \
    generation.positive_prompt=./examples/test_case.json \
    generation.output.dir=./output
