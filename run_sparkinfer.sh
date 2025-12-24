#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# export SPIF_SPLIT_DEBUG=ON
# export SPIF_DFR_DEBUG=1
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

model="/root/autodl-tmp/models/sparkinfer/prosparse-llama-13b.gguf"
model_split="/root/autodl-tmp/models/sparkinfer/prosparse-llama-2-13b-sparkinfer-model-split-768.gguf"

# model="/root/autodl-tmp/models/sparkinfer/prosparse-llama-13b-Q8_0.gguf"
# model_split="/root/autodl-tmp/models/sparkinfer/prosparse-llama-2-13b-sparkinfer-model-split-768.gguf"


prompt="Bubble sort algorithm in python: \n \`\`\`python"
# # prompt="# Dijkstra's shortest path algorithm in CPP (4 spaces indentation) + complexity analysis:\n\n"
prompt="Explain the theory of relativity in simple terms:\n\n"
# # prompt="Explain how the Large Language Model works to elementary school students (500 words). \n\n"
# prompt="Explain the concept of Large Language Models (LLMs) and their applications in natural language processing. \n\n"
prompt="Once upon a time,"
# prompt="Why aliens are not found in universe?\n\n"
# prompt="What is Artificial Intelligence? Please explain at least 300 words in Markdown format.\n\n"
# prompt="What is Artificial Intelligence? Write 300 words in Markdown, ensuring TWO line breaks between every paragraph.\n\n"

vram_budget=0
threads=12
seed=1234
ctx_size=1024
max_tokens=768
# n_prompts=20
# prompt_file="prompts.txt"

common_opts=(
    -spif-ms "$model_split"
    -cffn --no-mmap
    -vb "$vram_budget"
    -t "$threads"
    -s "$seed"
    -p "$prompt"
    -c "$ctx_size"
    -n "$max_tokens"
    --repeat-penalty 1.14
)

cli_opts=(
    -m "$model" -ngl 999 -no-cnv
)

speculative_opts=(
    -md "$draft_model" -m "$model"
    -ngld 999 -ngl 999 -kvu
    -co --draft-min 3 --draft-max 5
)

usage() {
    echo "usage: $0 [release|debug] [cli|speculative] [bench] [nvtx]"
    exit 1
}

mode=${1-}
kind=${2-}
bench_flag=0
nvtx_flag=0

extra_args=("${@:3}")
for arg in "${extra_args[@]}"; do
    case "$arg" in
    bench)
        bench_flag=1
        ;;
    nvtx)
        nvtx_flag=1
        ;;
    "") ;;
    *)
        usage
        ;;
    esac
done

bench_opts=()
if ((bench_flag)); then
    bench_opts=(-nps "$n_prompts" --file "$prompt_file")
fi

case "$mode" in
release)
    bin_dir="./build_rel/bin"
    ;;
debug)
    bin_dir="./build/bin"
    ;;
*)
    usage
    ;;
esac

case "$kind" in
cli)
    bin="$bin_dir/llama-cli"
    inference_opts=("${cli_opts[@]}" "${common_opts[@]}" "${bench_opts[@]}")
    ;;
speculative)
    bin="$bin_dir/llama-speculative"
    inference_opts=("${speculative_opts[@]}" "${common_opts[@]}" "${bench_opts[@]}")
    ;;
*)
    usage
    ;;
esac

if ((!nvtx_flag)); then
    "$bin" "${inference_opts[@]}"
else
    nsys profile --trace=cuda,nvtx "$bin" "${inference_opts[@]}"
fi
