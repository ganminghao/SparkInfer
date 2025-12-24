#!/bin/bash

MODE="$1"

NPS_LOW=10
NPS_HIGH=20
VB_LOW=12
VB_HIGH=24

export SPIF_DFR_DEBUG=2

mkdir -p dfr_logs

MODELS_SPIF_LOW=(
    opt-6.7b
    prosparse-llama-2-7b
    bamboo-7b
    sparseqwen2-7b
    opt-13b
    prosparse-llama-2-13b
)

MODELS_SPIF_HIGH=(
    opt-13b
    prosparse-llama-2-13b
    opt-30b
    relufalcon-40b
)

run_block() {
    local pwif_flag="$1"
    local cfg_file="$2"
    local log_dir="$3"
    local log_prefix="$4"
    local log_suffix="$5"
    shift 5
    local models=("$@")

    local n_prompts
    local vb_val
    if [[ $log_suffix == "low" ]]; then
        n_prompts="$NPS_LOW"
        vb_val="$VB_LOW"
    else
        n_prompts="$NPS_HIGH"
        vb_val="$VB_HIGH"
    fi

    if [[ -n $pwif_flag ]]; then
        bash compile_sparkinfer.sh release "$pwif_flag" nvtx
    else
        bash compile_sparkinfer.sh release nvtx
    fi

    for model in "${models[@]}"; do
        local nvtx_out="$(pwd)/${log_dir}/${log_prefix}_${model}_${log_suffix}_cli"
        bash run_eval.sh release cli \
            --cfg-file "$cfg_file" \
            --model-cfg "$model" \
            -vb "$vb_val" \
            -nps "$n_prompts" \
            nvtx="${nvtx_out}" \
            >"${log_dir}/${log_prefix}_${model}_${log_suffix}_cli.log" 2>&1
    done
}

if [[ -z $MODE || $MODE == "low" ]]; then
    run_block "" topk.yaml dfr_logs topk low "${MODELS_SPIF_LOW[@]}"
    run_block "" s_dfr.yaml dfr_logs s_dfr low "${MODELS_SPIF_LOW[@]}"
    run_block "" spif.yaml dfr_logs d_dfr low "${MODELS_SPIF_LOW[@]}"
fi

if [[ -z $MODE || $MODE == "high" ]]; then
    run_block "" topk.yaml dfr_logs topk high "${MODELS_SPIF_HIGH[@]}"
    run_block "" s_dfr.yaml dfr_logs s_dfr high "${MODELS_SPIF_HIGH[@]}"
    run_block "" spif.yaml dfr_logs d_dfr high "${MODELS_SPIF_HIGH[@]}"
fi
