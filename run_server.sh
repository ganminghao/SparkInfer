# do not modify these variables unless you know what you are doing
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

# you can modify the following variables
vram_budget=6
threads=12
seed=1234
ctx_size=2048
max_tokens=768
port=8080
# n_prompts=20
# prompt_file="prompts.txt"

opts=(
    -spif-ms "$model_split"
    -cffn --no-mmap
    -vb "$vram_budget"
    -t "$threads"
    -s "$seed"
    -c "$ctx_size"
)

./build/bin/llama-server -m "$model" "${opts[@]}" --port "$port"