#!/bin/bash

MODE="$1"

NPS_LOW=10
NPS_HIGH=20
VB_LOW=12
VB_HIGH=24

PROMPT_COUNT="$NPS_LOW"
if [[ $MODE == "high" || -z $MODE ]]; then
    PROMPT_COUNT="$NPS_HIGH"
fi

PROMPT_FILE="$PWD/prompts.txt"
if [[ ! -f $PROMPT_FILE ]]; then
    printf 'ERROR: prompt file not found: %s\n' "$PROMPT_FILE" >&2
    exit 1
fi

mapfile -t PROMPTS < <(head -n "$PROMPT_COUNT" "$PROMPT_FILE")

if [[ ${#PROMPTS[@]} -eq 0 ]]; then
    printf 'ERROR: no prompts found in %s\n' "$PROMPT_FILE" >&2
    exit 1
fi

mkdir -p tput_logs

MODELS_ORIG_LOW=(
    opt-6.7b
    prosparse-llama-2-7b
    bamboo-7b
    sparseqwen2-7b
    opt-13b
    prosparse-llama-2-13b
)
ORIG_LOW_START=40

MODELS_ORIG_HIGH=(
    opt-13b
    prosparse-llama-2-13b
    opt-30b
    relufalcon-40b
)
ORIG_HIGH_START=60

run_with_ngl_cli() {
    local ngl_start="$1"
    local n_prompts="$2"
    local vb_val="$3"
    local log_path="$4"
    local cfg_file="$5"
    local model="$6"

    local ngl
    for ((ngl = ngl_start; ngl >= 0; ngl--)); do
        if bash run_eval.sh release cli \
            --cfg-file "$cfg_file" \
            --model-cfg "$model" \
            -vb "$vb_val" \
            -nps "$n_prompts" \
            -ngl "$ngl" \
            bench >"$log_path" 2>&1; then
            return 0
        else
            rm -f "$log_path"
        fi
    done
    return 1
}

run_with_ngl_sd() {
    local ngl_start="$1"
    local vb_val="$2"
    local log_path="$3"
    local cfg_file="$4"
    local model="$5"
    local prompt="$6"

    local ngl
    for ((ngl = ngl_start; ngl >= 0; ngl--)); do
        if bash run_eval.sh release speculative \
            --cfg-file "$cfg_file" \
            --model-cfg "$model" \
            -vb "$vb_val" \
            -ngl "$ngl" \
            --prompt "$prompt" >"$log_path" 2>&1; then
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
        ngl_start="$ORIG_LOW_START"
        n_prompts="$NPS_LOW"
        vb_val="$VB_LOW"
    else
        ngl_start="$ORIG_HIGH_START"
        n_prompts="$NPS_HIGH"
        vb_val="$VB_HIGH"
    fi

    if [[ -n $pwif_flag ]]; then
        bash compile_sparkinfer.sh release "$pwif_flag"
    else
        bash compile_sparkinfer.sh release
    fi

    for model in "${models[@]}"; do
        local log_path_cli="${log_dir}/${log_prefix}_${model}_${log_suffix}_cli.log"
        run_with_ngl_cli "$ngl_start" "$n_prompts" "$vb_val" "$log_path_cli" "$cfg_file" "$model"
    done

    for ((i = 0; i < n_prompts && i < ${#PROMPTS[@]}; i++)); do
        local prompt="${PROMPTS[$i]}"
        for model in "${models[@]}"; do
            local log_path_sd="${log_dir}/${log_prefix}_${model}_${log_suffix}_sd-${i}.log"
            run_with_ngl_sd "$ngl_start" "$vb_val" "$log_path_sd" "$cfg_file" "$model" "$prompt"
        done
    done
}

if [[ -z $MODE || $MODE == "low" ]]; then
    run_block "" orig.yaml tput_logs orig low "${MODELS_ORIG_LOW[@]}"
fi

if [[ -z $MODE || $MODE == "high" ]]; then
    run_block "" orig.yaml tput_logs orig high "${MODELS_ORIG_HIGH[@]}"
fi
