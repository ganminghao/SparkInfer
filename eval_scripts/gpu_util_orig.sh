#!/bin/bash

MODE="$1"

NPS_LOW=10
NPS_HIGH=20
VB_LOW=12
VB_HIGH=24

mkdir -p gpu_util_logs

MODELS_SPIF_LOW=(
    sparseqwen2-7b
    opt-13b
)
GPU_LOW_START=40

MODELS_SPIF_HIGH=(
    prosparse-llama-2-13b
    relufalcon-40b
)
GPU_HIGH_START=60

run_with_ngl_nvtx() {
    local ngl_start="$1"
    local n_prompts="$2"
    local vb_val="$3"
    local log_path="$4"
    local nvtx_prefix="$5"
    local cfg_file="$6"
    local model="$7"

    local ngl
    for ((ngl = ngl_start; ngl >= 0; ngl--)); do
        if bash run_eval.sh release cli \
            --cfg-file "$cfg_file" \
            --model-cfg "$model" \
            -vb "$vb_val" \
            -ngl "$ngl" \
            -nps "$n_prompts" \
            nvtx="${nvtx_prefix}" \
            >"$log_path" 2>&1; then
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

    local ngl_start
    local n_prompts
    local vb_val
    if [[ $log_suffix == "low" ]]; then
        ngl_start="$GPU_LOW_START"
        n_prompts="$NPS_LOW"
        vb_val="$VB_LOW"
    else
        ngl_start="$GPU_HIGH_START"
        n_prompts="$NPS_HIGH"
        vb_val="$VB_HIGH"
    fi

    if [[ -n $pwif_flag ]]; then
        bash compile_sparkinfer.sh release "$pwif_flag"
    else
        bash compile_sparkinfer.sh release
    fi

    for model in "${models[@]}"; do
        local base="${log_dir}/${log_prefix}_${model}_${log_suffix}_cli"
        local log_path="${base}.log"
        run_with_ngl_nvtx "$ngl_start" "$n_prompts" "$vb_val" "$log_path" "$base" "$cfg_file" "$model"
    done
}

if [[ -z $MODE || $MODE == "low" ]]; then
    run_block "" orig.yaml gpu_util_logs orig low "${MODELS_SPIF_LOW[@]}"
fi

if [[ -z $MODE || $MODE == "high" ]]; then
    run_block "" orig.yaml gpu_util_logs orig high "${MODELS_SPIF_HIGH[@]}"
fi
