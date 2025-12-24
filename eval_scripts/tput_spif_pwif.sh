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

MODELS_SPIF_LOW=(
    opt-6.7b
    prosparse-llama-2-7b
    bamboo-7b
    sparseqwen2-7b
    opt-13b
    prosparse-llama-2-13b
)

MODELS_PWIF_LOW_1=(
    prosparse-llama-2-7b
    prosparse-llama-2-13b
    sparseqwen2-7b
)

MODELS_PWIF_LOW_2=(
    opt-6.7b
    opt-13b
)

MODELS_PWIF_LOW_3=(
    bamboo-7b
)

MODELS_SPIF_HIGH=(
    opt-13b
    prosparse-llama-2-13b
    opt-30b
    relufalcon-40b
)

MODELS_PWIF_HIGH_1=(
    prosparse-llama-2-13b
    relufalcon-40b
)

MODELS_PWIF_HIGH_2=(
    opt-13b
    opt-30b
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
            bench \
            >"${log_dir}/${log_prefix}_${model}_${log_suffix}_cli.log" 2>&1
    done

    for ((i = 0; i < n_prompts && i < ${#PROMPTS[@]}; i++)); do
        local prompt="${PROMPTS[$i]}"
        for model in "${models[@]}"; do
            bash run_eval.sh release speculative \
                --cfg-file "$cfg_file" \
                --model-cfg "$model" \
                -vb "$vb_val" \
                --prompt "$prompt" \
                >"${log_dir}/${log_prefix}_${model}_${log_suffix}_sd-${i}.log" 2>&1
        done
    done
}

if [[ -z $MODE || $MODE == "low" ]]; then
    run_block "" spif.yaml tput_logs spif low "${MODELS_SPIF_LOW[@]}"
    run_block "pwif=0.001" pwif.yaml tput_logs pwif low "${MODELS_PWIF_LOW_1[@]}"
    run_block "pwif=0.000001" pwif.yaml tput_logs pwif low "${MODELS_PWIF_LOW_2[@]}"
    run_block "pwif=0.3" pwif.yaml tput_logs pwif low "${MODELS_PWIF_LOW_3[@]}"
fi

if [[ -z $MODE || $MODE == "high" ]]; then
    run_block "" spif.yaml tput_logs spif high "${MODELS_SPIF_HIGH[@]}"
    run_block "pwif=0.001" pwif.yaml tput_logs pwif high "${MODELS_PWIF_HIGH_1[@]}"
    run_block "pwif=0.000001" pwif.yaml tput_logs pwif high "${MODELS_PWIF_HIGH_2[@]}"
fi
