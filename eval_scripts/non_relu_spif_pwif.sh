#!/bin/bash

MODE="$1"

NPS_LOW=10
NPS_HIGH=20
VB_LOW=12
VB_HIGH=24

mkdir -p non_relu_logs

MODELS_LOW=(
    prosparse-llama-2-13b
    sparseqwen2-7b
)

MODELS_HIGH=(
    prosparse-llama-2-13b
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
        bash compile_sparkinfer.sh release "$pwif_flag"
    else
        bash compile_sparkinfer.sh release
    fi

    for model in "${models[@]}"; do
        bash run_eval.sh release cli \
            --cfg-file "$cfg_file" \
            --model-cfg "$model" \
            -vb "$vb_val" \
            -nps "$n_prompts" \
            bench >"${log_dir}/${log_prefix}_${model}_${log_suffix}_cli.log" 2>&1
    done
}

if [[ -z $MODE || $MODE == "low" ]]; then
    run_block "pwif=0.0001" spif.yaml non_relu_logs non_relu_spif low "${MODELS_LOW[@]}"
    run_block "pwif=0.0" pwif.yaml non_relu_logs non_relu_pwif low "${MODELS_LOW[@]}"
fi

if [[ -z $MODE || $MODE == "high" ]]; then
    run_block "pwif=0.0001" spif.yaml non_relu_logs non_relu_spif high "${MODELS_HIGH[@]}"
    run_block "pwif=0.0" pwif.yaml non_relu_logs non_relu_pwif high "${MODELS_HIGH[@]}"
fi
