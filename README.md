![output](/render1.png)

# Raytracing with CUDA
This is a program that implements a simple raytracing engine in CUDA C++ and SDL.

## Requirements
- SDL2
- NVCC
- Nvidia GPU with CUDA support

## Configuration and Installation
The resolution of the output window can be configured before compilation by changing the ```#WIDTH``` and ```#HEIGHT``` constants.  
Build using ```nvcc rt_cuda.cu -lSDL2 -lm ```

## Usage
Run ```./a.out```  
The camera can be controlled using
W/S to move forward/back, A/D to strafe, Q/E to turn, and Z/X to move up/down.
