#!/bin/bash

set -x
export CUDA_VISIBLE_DEVICES=0

CFG_DIR="$(pwd)/cfgs"
DEFAULT_CFG_FILE="${CFG_DIR}/default.yaml"
MODELS_YAML="${MODELS_YAML:-$DEFAULT_CFG_FILE}"

usage() {
    cat <<EOF
usage: $0 [release|debug] [cli|speculative] [options...]

options (only):
  --cfg-file FILE         YAML file under ./cfgs/ (e.g. my_models.yaml)
  --model-cfg NAME        Model config name defined in YAML
  --prompt TEXT           Override prompt text
  -vb N                   Override vram_budget (VRAM budget)
  -ngl N                  Override ngl
  -nps N                  Override n_prompts (number of prompts for bench)
  bench                   Enable benchmark mode
  nvtx=NAME               Profile with nsys, output name is NAME

Notes:
  - All model paths and numeric parameters must be configured in YAML.
  - Default YAML file: ./cfgs/default.yaml
  - In speculative mode, draft_model MUST be set in YAML, otherwise it is an error.
EOF
    exit 1
}

need_yq() {
    if ! command -v yq >/dev/null 2>&1; then
        echo "ERROR: yq is not installed (required for YAML parsing)." >&2
        exit 1
    fi
    if [[ ! -f $MODELS_YAML ]]; then
        echo "ERROR: YAML config not found: $MODELS_YAML" >&2
        exit 1
    fi
}

yaml_get() {
    local name="$1"
    local key="$2"
    yq eval -r ".\"${name}\".${key} // \"null\"" "$MODELS_YAML"
}

yaml_with_default() {
    local model="$1"
    local key="$2"
    local v
    v=$(yaml_get "$model" "$key")
    if [[ $v == "null" ]]; then
        yaml_get "default" "$key"
    else
        echo "$v"
    fi
}

model=""
model_split=""
draft_model=""

ngl=999
vram_budget=0
threads=12
seed=42
ctx_size=768
max_tokens=512
n_prompts=10
prompt_file="prompts.txt"
prompt="Bubble sort algorithm in python:"

cffn_flag=1
no_mmap_flag=1

# CLI override flags: if set, YAML should not overwrite
vb_override=0
ngl_override=0
nps_override=0

# nvtx output file name (empty means no profiling)
nvtx_out=""

MODEL_CFG_ENV="${MODEL_CFG:-}"
MODEL_CFG_CLI=""

apply_spif_env() {
    local name="$1"

    local keys=(
        dfr_ema
        dx_dfr_decay
        init_dfr_decay
        parallel
        reload
        reload_window_size
        reorder
        split_debug
    )

    for k in "${keys[@]}"; do
        local val
        val=$(yaml_with_default "$name" "spif.${k}")
        # treat empty or "null" as "do not export"
        if [[ -n $val && $val != "null" ]]; then
            local env_key="SPIF_$(echo "$k" | tr 'a-z' 'A-Z')"
            export "${env_key}=${val}"
        fi
    done
}

load_model_cfg() {
    local name="$1"
    need_yq

    local v

    v=$(yaml_with_default "$name" "model")
    if [[ $v == "null" ]]; then
        echo "ERROR: 'model' field missing for config '${name}'." >&2
        exit 1
    fi
    model="$v"

    v=$(yaml_with_default "$name" "model_split")
    [[ $v != "null" ]] && model_split="$v" || model_split=""

    v=$(yaml_with_default "$name" "draft_model")
    [[ $v != "null" ]] && draft_model="$v" || draft_model=""

    v=$(yaml_with_default "$name" "ngl")
    if [[ $v != "null" && $ngl_override -eq 0 ]]; then
        ngl="$v"
    fi

    v=$(yaml_with_default "$name" "vram_budget")
    if [[ $v != "null" && $vb_override -eq 0 ]]; then
        vram_budget="$v"
    fi

    v=$(yaml_with_default "$name" "threads")
    [[ $v != "null" ]] && threads="$v"

    v=$(yaml_with_default "$name" "ctx_size")
    [[ $v != "null" ]] && ctx_size="$v"

    v=$(yaml_with_default "$name" "max_tokens")
    [[ $v != "null" ]] && max_tokens="$v"

    v=$(yaml_with_default "$name" "n_prompts")
    if [[ $v != "null" && $nps_override -eq 0 ]]; then
        n_prompts="$v"
    fi

    v=$(yaml_with_default "$name" "prompt_file")
    [[ $v != "null" ]] && prompt_file="$v"

    v=$(yaml_with_default "$name" "cffn")
    [[ $v != "null" ]] && cffn_flag="$v"

    v=$(yaml_with_default "$name" "no_mmap")
    [[ $v != "null" ]] && no_mmap_flag="$v"

    apply_spif_env "$name"
}

mode=${1-}
kind=${2-}
shift 2 || true

[[ $mode == "release" || $mode == "debug" ]] || usage
[[ $kind == "cli" || $kind == "speculative" ]] || usage

bench_flag=0

while [[ $# -gt 0 ]]; do
    case "$1" in
    --cfg-file)
        MODELS_YAML="$CFG_DIR/$2"
        shift 2
        ;;
    --model-cfg)
        MODEL_CFG_CLI="$2"
        shift 2
        ;;
    --prompt)
        prompt="$2"
        shift 2
        ;;
    -vb)
        vram_budget="$2"
        vb_override=1
        shift 2
        ;;
    -ngl)
        ngl="$2"
        ngl_override=1
        shift 2
        ;;
    -nps)
        n_prompts="$2"
        nps_override=1
        shift 2
        ;;
    bench)
        bench_flag=1
        shift
        ;;
    nvtx=*)
        nvtx_out="${1#nvtx=}"
        shift
        ;;
    *)
        echo "Unknown option: $1" >&2
        usage
        ;;
    esac
done

if [[ -n $MODEL_CFG_CLI ]]; then
    load_model_cfg "$MODEL_CFG_CLI"
elif [[ -z $model && -n $MODEL_CFG_ENV ]]; then
    load_model_cfg "$MODEL_CFG_ENV"
fi

if [[ -z $model ]]; then
    echo "ERROR: No model specified. Use --model-cfg or set MODEL_CFG." >&2
    usage
fi

common_opts=(
    -ngl "$ngl"
    -vb "$vram_budget"
    -t "$threads"
    -s "$seed"
    -p "$prompt"
    -c "$ctx_size"
    -n "$max_tokens"
)

if [[ $cffn_flag == "1" ]]; then
    common_opts=(-cffn "${common_opts[@]}")
fi

if [[ $no_mmap_flag == "1" ]]; then
    common_opts=(--no-mmap "${common_opts[@]}")
fi

if [[ -n $model_split ]]; then
    common_opts=(-spif-ms "$model_split" "${common_opts[@]}")
fi

cli_opts=(
    -m "$model"
    -no-cnv
    --repeat-penalty 1.1
)

speculative_opts=()
if [[ -n $draft_model ]]; then
    speculative_opts=(
        -m "$model"
        -md "$draft_model"
        -ngld 999
        --draft-min 3
        --draft-max 5
        --repeat-penalty 1.2
    )
fi

bench_opts=()
if ((bench_flag)); then
    bench_opts=(-nps "$n_prompts" --file "$prompt_file")
fi

case "$mode" in
release) bin_dir="./build_rel/bin" ;;
debug) bin_dir="./build/bin" ;;
esac

case "$kind" in
cli)
    bin="$bin_dir/llama-cli"
    inference_opts=("${cli_opts[@]}" "${common_opts[@]}" "${bench_opts[@]}")
    ;;
speculative)
    if [[ ${#speculative_opts[@]} -eq 0 ]]; then
        echo "ERROR: speculative mode requires a draft_model configured in YAML." >&2
        exit 1
    fi
    bin="$bin_dir/llama-speculative"
    inference_opts=("${speculative_opts[@]}" "${common_opts[@]}")
    ;;
esac

if [[ -n $nvtx_out ]]; then
    nsys profile -o "$nvtx_out" --force-overwrite=true --trace=cuda,nvtx "$bin" "${inference_opts[@]}"
else
    "$bin" "${inference_opts[@]}"
fi
