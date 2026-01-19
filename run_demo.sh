#!/bin/bash

# SparkInfer environment variables, should not be modified unless you know what you are doing
export SPIF_DFR_EMA=ON
export SPIF_INIT_DFR_DECAY=67
export SPIF_DX_DFR_DECAY=51
export SPIF_RELOAD_WINDOW_SIZE=4
export SPIF_REORDER=ON
export SPIF_PARALLEL=ON
export SPIF_RELOAD=ON

# choose model and model-split file
model=""
model_split=""

# set up prompts
prompt="Once upon a time,"

# you can modify the following variables
# hardware rescources settings
vram_budget=0  # 0 means using all available VRAM
threads=12

# generation settings
seed=1234
ctx_size=1024
max_tokens=768

opts=(
    -m "$model" 
    -spif-ms "$model_split"
    -ngl 999 -no-cnv
    -cffn --no-mmap
    -vb "$vram_budget"
    -t "$threads"
    -s "$seed"
    -p "$prompt"
    -c "$ctx_size"
    -n "$max_tokens"
    --repeat-penalty 1.14
)

# run the model with SparkInfer
./build/bin/llama-cli "${opts[@]}"
