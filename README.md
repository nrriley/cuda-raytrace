![output](/render1.png)

# Raytracing with Cuda
This is a program that implements a simple raytracing engine in CUDA and SDL

## Requirements
- SDL2
- nvcc

## Configuration and Installation
The resolution of the output window can be configured before compilation by changing the ```#WIDTH``` and ```#HEIGHT``` constants.
Build using ```nvcc rt_cuda.cu -lSDL2 -lm ```
