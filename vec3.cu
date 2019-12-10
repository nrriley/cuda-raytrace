#include <cuda.h>
typedef uchar4 Color; // .x->R, .y->G, .z->B, .w->A
__device__ double3 operator+ (double3 v1, double3 v2)
{
    double3 retval;
    retval.x = v1.x+v2.x;
    retval.y = v1.y+v2.y;
    retval.z = v1.z+v2.z;
    return retval;
}

__device__ double3 operator- (double3 v1, double3 v2)
{
    double3 retval;
    retval.x = v1.x-v2.x;
    retval.y = v1.y-v2.y;
    retval.z = v1.z-v2.z;
    return retval;
}

__device__ double3 operator* (double3 v1, double3 v2)
{
    double3 retval;
    retval.x = v1.x*v2.x;
    retval.y = v1.y*v2.y;
    retval.z = v1.z*v2.z;
    return retval;
}

__device__ double3 operator/ (double3 v1, double3 v2)
{
    double3 retval;
    retval.x = v1.x/v2.x;
    retval.y = v1.y/v2.y;
    retval.z = v1.z/v2.z;
    return retval;
}

__device__ double3 operator* (double scalar, double3 vec)
{
    double3 retval;
    retval.x = scalar*vec.x;
    retval.y = scalar*vec.y;
    retval.z = scalar*vec.z;
    return retval;
}

__device__ Color operator* (double scalar, Color vec)
{
    Color retval;
    retval.x = scalar*vec.x;
    retval.y = scalar*vec.y;
    retval.z = scalar*vec.z;
    return retval;
}

__device__ Color operator+ (Color vec1, Color vec2)
{
    Color retval = vec1;
    retval.x = vec1.x+vec2.x;
    retval.y = vec1.y+vec2.y;
    retval.z = vec1.z+vec2.z;
    return retval;
}
