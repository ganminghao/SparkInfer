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

NGL_LOW_START=40
NGL_HIGH_START=60

run_with_ngl_vb() {
    local ngl_start="$1"
    local n_prompts="$2"
    local log_path="$3"
    local cfg_file="$4"
    local model="$5"
    local vb="$6"

    local ngl
    for ((ngl = ngl_start; ngl >= 0; ngl--)); do
        if bash run_eval.sh release cli \
            --cfg-file "$cfg_file" \
            --model-cfg "$model" \
            -ngl "$ngl" \
            -vb "$vb" \
            -nps "$n_prompts" \
            bench >"$log_path" 2>&1; then
            return 0
        else
            rm -f "$log_path"
        fi
    done
    return 1
}

run_block() {
    local pwif_flag="$1"
    local cfg_file="$2"
    local log_dir="$3"
    local log_prefix="$4"
    local log_suffix="$5"
    shift 5
    local models=("$@")

    local vbs=()
    local ngl_start
    local n_prompts

    if [[ $log_suffix == "low" ]]; then
        vbs=("${VB_LOW[@]}")
        ngl_start="$NGL_LOW_START"
        n_prompts="$NPS_LOW"
    else
        vbs=("${VB_HIGH[@]}")
        ngl_start="$NGL_HIGH_START"
        n_prompts="$NPS_HIGH"
    fi

    if [[ -n $pwif_flag ]]; then
        bash compile_sparkinfer.sh release "$pwif_flag"
    else
        bash compile_sparkinfer.sh release
    fi

    for vb in "${vbs[@]}"; do
        for model in "${models[@]}"; do
            local log_path="${log_dir}/${log_prefix}_${vb}_${model}_${log_suffix}_cli.log"
            run_with_ngl_vb "$ngl_start" "$n_prompts" "$log_path" "$cfg_file" "$model" "$vb"
        done
    done
}

if [[ -z $MODE || $MODE == "low" ]]; then
    run_block "" orig.yaml vb_logs orig_vb low "${MODELS_LOW[@]}"
fi

if [[ -z $MODE || $MODE == "high" ]]; then
    run_block "" orig.yaml vb_logs orig_vb high "${MODELS_HIGH[@]}"
fi
