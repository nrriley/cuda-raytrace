![output](/render1.png)

# Raytracing with CUDA
This is a program that implements a simple raytracing engine in CUDA C++ and SDL.

## Requirements
- SDL2
- NVCC
- Nvidia GPU with CUDA support

## Configuration and Installation
The resolution of the output window can be configured before compilation by changing the ```#WIDTH``` and ```#HEIGHT``` constants.  
The number of reflections per sample is set by ```#RT_DEPTH```.  
The number of samples per pixel in pathtracing mode is set by ```#RAYS_PER_PIXEL```.  
Build using ```nvcc rt_cuda.cu -lSDL2 -lm ```  

## Usage
Run ```./a.out```  
The camera can be controlled using
W/S to move forward/back, A/D to strafe, Q/E to turn, and Z/X to move up/down.
Press R to toggle pathtracing. (disabled by default)

## Pathtracing Modes
Pathtracing is disabled initially in favor of raytracing, so that the camera can be positioned more easily. To toggle pathtracing, press R.  

Two types of pathtracing are currently implemented:  
The first type is a method where reflected rays are created by first calculating the perfect reflected ray, and then modifying it by adding a small vector who's component magnitudes fall within a range determined my the roughness value of the surface material.  
The second type simply randomly chooses a ray in the same hemisphere as the normal of the surface at the reflection point. This method is only able to model diffuse reflection, and larger values of ```#RAYS_PER_PIXEL``` are needed.  

*Mode One Pathtracing*  

The first method is enabled by default. To use the second method instead, build with:       ```nvcc rt_cuda.cu -lSDL2 -lm -DPT_NAIVE```

In the future I will implement pathtracing with GGX Importance Sampling.

## Benchmarking
A non-interactive benchmarking mode is included. Build with ```nvcc rt_cuda.cu -lSDL2 -lm -DBENCH=N```, where N is the total number of frames to render. To enable pathtracing by default for benchmarking, build with ```nvcc rt_cuda.cu -lSDL2 -lm -DBENCH=N -DPT_ON```

## Scene Definition
The scene is initialized in the scene_setup() function. A sample scene is provided. Currently, spheres and triangles are supported. Ambient, Point, and Directional lights are supported. See the provided sample scene for examples on how to add each type of object to the scene.

ASCII STL files can be loaded using the AddTrianglesFromSTL() function, which supports an optional position offset and scaling factor.
