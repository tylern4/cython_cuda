# cython_cuda

Seeing if I can make cuda code work in python using cython wrappers. 

Mostly for me to look back at if I want to compute something with GPU/CUDA in the future to see what I need to do.


## Building

```bash
mkdir -p cython_cuda/build
cd cython_cuda/build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=52 -DCMAKE_BUILD_TYPE=Release
make
```


