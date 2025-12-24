#!/bin/bash

MODE="$1"

NPS_LOW=10
NPS_HIGH=20

mkdir -p vb_logs

MODELS_LOW=(
    bamboo-7b
)

MODELS_HIGH=(
    opt-13b
)

VB_LOW=(7 8 9 10 11 12)
VB_HIGH=(14 16 18 20 22 24)

run_block() {
    local pwif_flag="$1"
    local cfg_file="$2"
    local log_dir="$3"
    local log_prefix="$4"
    local log_suffix="$5"
    shift 5
    local models=("$@")

    local vbs=()
    local n_prompts
    if [[ $log_suffix == "low" ]]; then
        vbs=("${VB_LOW[@]}")
        n_prompts="$NPS_LOW"
    else
        vbs=("${VB_HIGH[@]}")
        n_prompts="$NPS_HIGH"
    fi

    if [[ -n $pwif_flag ]]; then
        bash compile_sparkinfer.sh release "$pwif_flag"
    else
        bash compile_sparkinfer.sh release
    fi

    for vb in "${vbs[@]}"; do
        for model in "${models[@]}"; do
            bash run_eval.sh release cli \
                --cfg-file "$cfg_file" \
                --model-cfg "$model" \
                -vb "$vb" \
                -nps "$n_prompts" \
                bench >"${log_dir}/${log_prefix}_${vb}_${model}_${log_suffix}_cli.log" 2>&1
        done
    done
}

if [[ -z $MODE || $MODE == "low" ]]; then
    run_block "" spif.yaml vb_logs spif_vb low "${MODELS_LOW[@]}"
    run_block "pwif=0.3" pwif.yaml vb_logs pwif_vb low "${MODELS_LOW[@]}"
fi

if [[ -z $MODE || $MODE == "high" ]]; then
    run_block "" spif.yaml vb_logs spif_vb high "${MODELS_HIGH[@]}"
    run_block "pwif=0.000001" pwif.yaml vb_logs pwif_vb high "${MODELS_HIGH[@]}"
fi
