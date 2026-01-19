# SparkInfer

Fastest local LLM inference by activation sparsity

## Demo Video
<video width="640" height="360" controls>
    <source src="figures/demo.mp4" type="video/mp4">
</video>
SparkInfer v.s. PowerInfer on a single RTX 4090(24GB) running ProSparse-Llama-2-13B FP16(26GB) with 2 times speedup!

## Abstract
We present SparkInfer, an adaptive GPU–CPU hybrid inference system that addresses these limitations through online neuron balancing, a mechanism that dynamically redistributes neurons between the GPU and CPU based on activation behaviors. Extensive evaluations on consumer-grade PCs demonstrate that SparkInfer improves end-to-end throughput by up to 5.05×, 2.48×, and 3.71× over llama.cpp, PowerInfer, and Neuralink, respectively.


## Models Weights
### Supported Models
// TODO
### Download Models
// TODO



## Getting Started with SparkInfer

### Compilation Instructions
To compile SparkInfer, follow these steps:
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_CUDA_GRAPHS=OFF
cmake --build build --config Release -j"$(nproc)" --target llama-cli
```

**Optional Compilation Speed-Up**: You can specify `-DCMAKE_CUDA_ARCHITECTURES=""` to reduce compilation time by targeting only your NVIDIA GPU architecture. Replace the value with the appropriate architecture for your GPU. For example:
- For RTX 4090: `-DCMAKE_CUDA_ARCHITECTURES="89-real"`

### Running a Demo
1. Update the model path (and model-split path) and configure the desired hardware settings in `run_demo.sh`.
2. Execute the demo script:
```bash
bash run_demo.sh
```


## Evaluation
We evaluated SparkInfer against llama.cpp, PowerInfer, and Neuralink on two PC configurations: **PC-Low** (NVIDIA RTX 3080Ti, 12GB) and **PC-High** (NVIDIA RTX 4090, 24GB). The results below demonstrate that SparkInfer achieves significant performance improvements, with up to **5.05× speedup** over llama.cpp, **3.71×** over Neuralink, and **2.48×** over PowerInfer.

![tput](figures/tput.png)

More details could be find in our paper.

## Paper and Citation
// TODO
