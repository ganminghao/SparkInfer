# a demo compile script for sparkinfer with 4090
cmake -B build -DCMAKE_BUILD_TYPE=Release  /
    -DGGML_CUDA=ON /
    -DBUILD_SHARED_LIBS=OFF /
    -DGGML_CUDA_GRAPHS=OFF /
    -DCMAKE_CUDA_ARCHITECTURES="89-real"
cmake --build build --config Release -j"$(nproc)" --target llama-cli