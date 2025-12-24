export SPIF_DFR_EMA=ON
export SPIF_INIT_DFR_DECAY=67
export SPIF_DX_DFR_DECAY=51
export SPIF_RELOAD_WINDOW_SIZE=4

export SPIF_REORDER=ON
export SPIF_PARALLEL=ON
export SPIF_RELOAD=ON

# model="/root/autodl-tmp/models/sparkinfer/opt-6.7b.gguf"
# model_split="/root/autodl-tmp/models/sparkinfer/opt-6.7b-sparkinfer-model-split-1024.gguf"
# draft_model="/root/autodl-tmp/models/sparkinfer/opt-125m.gguf"


# model="/root/autodl-tmp/models/sparkinfer/Bamboo-7b.gguf"
# model_split="/root/autodl-tmp/models/sparkinfer/bamboo-7b-sparkinfer-model-split-896.gguf"
# draft_model="/root/autodl-tmp/models/sparkinfer/lite-mistral-150m.gguf"

model="/root/autodl-tmp/models/sparkinfer/SparseQwen2-7b.gguf"
model_split="/root/autodl-tmp/models/sparkinfer/SparseQwen2-7B-sparkinfer-model-split-592.gguf"
# # draft_model="/root/autodl-tmp/models/sparkinfer/Qwen2-0.5B.gguf"
# draft_model="/root/autodl-tmp/models/sparkinfer/Qwen-200m.gguf"
# # draft_model="/root/autodl-tmp/models/sparkinfer/Qwen2-0.5b-q8.gguf"
# model="/root/autodl-tmp/models/sparkinfer/Qwen2-0.5B.gguf"

# model="/root/autodl-tmp/models/sparkinfer/prosparse-llama-7b.gguf"
# model_split="/root/autodl-tmp/models/sparkinfer/prosparse-llama-2-7b-sparkinfer-model-split-688.gguf"
# model="/root/autodl-tmp/models/sparkinfer/prosparse-7b-vp.gguf"
# # draft_model="/root/autodl-tmp/models/gguf-Llama-160M-Chat-v1/Llama-160M-Chat-v1.F16.gguf"

# model_split="/root/autodl-tmp/models/sparkinfer/test.gguf"
# # # model_split="/root/autodl-tmp/models/sparkinfer/prosparse-llama-2-7b-sparkinfer-model-split.gguf"
# # model_split="/root/autodl-tmp/models/spa÷rkinfer/prosparse-llama-2-7b-sparkinfer-model-split-344.gguf"
# model_split="/root/autodl-tmp/models/sparkinfer/prosparse-llama-2-7b-predictor-688-sparkinfer-model-split.gguf"

# model="/root/autodl-tmp/models/sparkinfer/opt-13b.gguf"
# model_split="/root/autodl-tmp/models/sparkinfer/opt-13b-sparkinfer-model-split-1024.gguf"

# model="/root/autodl-tmp/models/sparkinfer/prosparse-llama-13b.gguf"
# model_split="/root/autodl-tmp/models/sparkinfer/prosparse-llama-2-13b-sparkinfer-model-split-768.gguf"

# model="/root/autodl-tmp/models/sparkinfer/prosparse-llama-13b-Q8_0.gguf"
# model_split="/root/autodl-tmp/models/sparkinfer/prosparse-llama-2-13b-sparkinfer-model-split-768.gguf"

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

./build_rel/bin/llama-server -m "$model" "${opts[@]}" --port "$port"