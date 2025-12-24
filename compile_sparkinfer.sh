#!/bin/bash

if ! dpkg -s libcurl4-gnutls-dev >/dev/null 2>&1; then
    apt update >/dev/null 2>&1
    apt install -y libcurl4-gnutls-dev >/dev/null 2>&1
fi

if ! command -v yq >/dev/null 2>&1; then
    latest_tag=$(
        curl -s https://api.github.com/repos/mikefarah/yq/releases/latest |
            grep '"tag_name"' | sed -E 's/.*"([^"]+)".*/\1/'
    )
    url="https://github.com/mikefarah/yq/releases/download/${latest_tag}/yq_linux_amd64"
    wget -q "$url" -O /usr/bin/yq
    chmod a+x /usr/bin/yq
fi

release_dir=build_rel
debug_dir=build

cmake_opts=(
    -DBUILD_SHARED_LIBS=OFF
    -DGGML_CUDA=ON
    -DGGML_CUDA_GRAPHS=OFF
    -DCMAKE_CUDA_ARCHITECTURES="89-real"
)

usage() {
    echo "usage: $0 [release|debug] [pwif|pwif=0.xx] [nvtx]"
    exit 1
}

mode=${1-}
shift || true

pwif_flag=0
pwif_val="0.10"
nvtx_flag=0

extra_args=("${@-}")
for arg in "${extra_args[@]}"; do
    case "$arg" in
    nvtx)
        nvtx_flag=1
        ;;
    pwif)
        pwif_flag=1
        pwif_val="0.10"
        ;;
    pwif=*)
        pwif_flag=1
        pwif_val="${arg#pwif=}"
        ;;
    "") ;;
    *)
        usage
        ;;
    esac
done

case "$mode" in
release)
    build_dir="$release_dir"
    build_type=Release
    ;;
debug)
    build_dir="$debug_dir"
    build_type=Debug
    cmake_opts+=(-DGGML_CUDA_DEBUG=ON)
    ;;
*)
    usage
    ;;
esac

extra_flags=()

((nvtx_flag)) && extra_flags+=(-DUSE_NVTX -I/usr/local/cuda/include)
((pwif_flag)) && extra_flags+=(-DSPIF_SPARSE_THRESHOLD="${pwif_val}f")

if ((${#extra_flags[@]} > 0)); then
    joined="${extra_flags[*]}"
    cmake_opts+=(
        -DCMAKE_C_FLAGS="$joined"
        -DCMAKE_CXX_FLAGS="$joined"
        -DCMAKE_CUDA_FLAGS="$joined"
    )
fi

flags_id="mode=${mode} pwif=${pwif_flag}:${pwif_val} nvtx=${nvtx_flag}"
stamp_file="${build_dir}/.build_flags"

if [[ -d $build_dir ]]; then
    if [[ -f $stamp_file ]]; then
        last=$(<"$stamp_file")
        if [[ $last == "$flags_id" ]]; then
            echo "[compile] reuse existing build: $build_dir (${flags_id})"
            exit 0
        else
            echo "[compile] flags changed, clean build_dir: $build_dir"
            rm -rf "$build_dir"
        fi
    else
        echo "[compile] no stamp found, clean build_dir: $build_dir"
        rm -rf "$build_dir"
    fi
fi

cmake -B "$build_dir" -DCMAKE_BUILD_TYPE="$build_type" "${cmake_opts[@]}"
cmake --build "$build_dir" --config "$build_type" -j"$(nproc)" \
    --target llama-cli llama-speculative llama-quantize llama-server

mkdir -p "$build_dir"
echo "$flags_id" >"$stamp_file"
