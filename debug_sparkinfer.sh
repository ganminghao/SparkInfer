#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# export SPIF_SPLIT_DEBUG=ON
# export SPIF_DFR_DEBUG=1
# export SPIF_DFR_EMA=ON
export SPIF_INIT_DFR_DECAY=67
export SPIF_DX_DFR_DECAY=20
export SPIF_RELOAD_WINDOW_SIZE=4

export SPIF_REORDER=ON
export SPIF_PARALLEL=ON
export SPIF_RELOAD=ON

# draft_model="/share/models/sparkinfer-models/opt-125m.gguf"
# model="/share/models/sparkinfer-models/opt-6.7b.gguf"
# model_split="/share/models/sparkinfer-models/opt-6.7b-sparkinfer-model-split-1024.gguf"
# model="/share/models/sparkinfer-models/opt-13b-q8_0.gguf"
# model_split="/share/models/sparkinfer-models/opt-13b-sparkinfer-model-split-1024.gguf"
# model="/share/models/sparkinfer-models/opt-30b-q8_0.gguf"
# model_split="/share/models/sparkinfer-models/opt-30b-sparkinfer-model-split-1024.gguf"
# draft_model="/share/models/sparkinfer-models/llama-160m-chat.gguf"
# model="/share/models/sparkinfer-models/prosparse-llama-2-7b.gguf"
# model_split="/share/models/sparkinfer-models/prosparse-llama-2-7b-sparkinfer-model-split-688.gguf"
# model="/share/models/sparkinfer-models/prosparse-llama-2-13b-q8_0.gguf"
# model_split="/share/models/sparkinfer-models/prosparse-llama-2-13b-sparkinfer-model-split-864.gguf"
# draft_model="/share/models/sparkinfer-models/lite-mistral-150m.gguf"
# model="/share/models/sparkinfer-models/bamboo-7b.gguf"
# model_split="/share/models/sparkinfer-models/Bamboo-base-v0_1-sparkinfer-model-split-896.gguf"
# draft_model="/share/models/sparkinfer-models/Qwen2-0.5B.gguf"
# model="/share/models/sparkinfer-models/SparseQwen2-7B.gguf"
# model_split="/share/models/sparkinfer-models/SparseQwen2-7B-sparkinfer-model-split-592.gguf"
prompt="Bubble sort algorithm in python:"

vram_budget=10
threads=8
seed=1234
ctx_size=1024
max_tokens=128
n_prompts=20
prompt_file="prompts.txt"

common_opts=(
    -spif-ms "$model_split"
    -cffn --no-mmap
    -vb "$vram_budget"
    -t "$threads"
    -s "$seed"
    -p "$prompt"
    -c "$ctx_size"
    -n "$max_tokens"
)

cli_opts=(
    -m "$model" -ngl 999 -no-cnv
)

speculative_opts=(
    -md "$draft_model" -m "$model"
    -ngld 999 -ngl 999 -kvu
    -co --draft-min 3 --draft-max 5
    # --repeat-penalty 1.1
)

usage() {
    echo "usage: $0 [cli|speculative] [bench] [cuda]"
    exit 1
}

kind=${1-}
bench_flag=0
cuda_flag=0

extra_args=("${@:2}")
for arg in "${extra_args[@]}"; do
    case "$arg" in
    bench)
        bench_flag=1
        ;;
    cuda)
        cuda_flag=1
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

case "$kind" in
cli)
    bin="./build/bin/llama-cli"
    inference_opts=("${cli_opts[@]}" "${common_opts[@]}" "${bench_opts[@]}")
    ;;
speculative)
    bin="./build/bin/llama-speculative"
    inference_opts=("${speculative_opts[@]}" "${common_opts[@]}" "${bench_opts[@]}")
    ;;
*)
    usage
    ;;
esac

if ((!cuda_flag)); then
    dbg=gdb
else
    dbg=cuda-gdb
fi

$dbg --args "$bin" "${inference_opts[@]}"
