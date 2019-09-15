#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <float.h>
#include <cuda.h>
#include "vec3.cu"

#define WIDTH 1280
#define HEIGHT 720
#define BLOCKX 16
#define BLOCKY 16
#define BG_COLOR {150, 150, 255, 255}
#define MS .1
#define TR .05
#define RT_DEPTH 3

// Typedefs
typedef uchar4 Color; // .x->R, .y->G, .z->B, .w->A
typedef enum Light_Type {
    AMBIENT,
    DIRECTIONAL,
    POINT
} Light_Type;

typedef struct Sphere {
    double3 center;
    double radius;
    double specular;
    double reflectivity;
    Color color;
} Sphere;

typedef struct Light {
    double intensity;
    Light_Type type;
    union {
        double3 pos;
        double3 dir;
    };
} Light;

typedef struct Scene {
    Sphere **spheres;
    int sphere_n;
    Light **lights;
    int light_n;
} Scene;

// Host Functions
void output_fb_to_sdl(SDL_Renderer *renderer, Color *framebuffer, int width, int height);
void AddSphere(int radius, double3 center, Color color, double specular, double reflect);
void AddLight(Light_Type type, double intensity);
void AddLight(Light_Type type, double intensity, double3 pos_dir);
// Global Functions
__global__ void renderSingleFrame(Color *framebuffer, int width, int height);

// Device Functions
__device__ double3 CanvasToViewport(int x, int y, int width, int height);
__device__  Color TraceRay(double3 O, double3 viewport, double t_min, double t_max, Color bg_color, int depth);
__device__ double2 IntersectRaySphere(double3 O, double3 viewport, Sphere *sphere);
__device__ double ComputeLighting(double3 point, double3 normal, double3 view, double spec);
__device__ double ClosestIntersection(double3 O, double3 viewport, double t_min, double t_max, int *sphere_index);
__device__ double3 ReflectRay(double3 R, double3 normal);
// linear algebra
__device__ double dot(double3 vec1, double3 vec2);
__device__ double length(double3 vec);

// Global variables
__managed__ double3 origin = {0.0, 0.0, 0.0}; // Current camera position
__managed__ double theta = 0.0; // Rotation of camera about Y axis
__managed__ Scene scene; // struct containing objects to be rendered


int main(int argc, char const *argv[])
{
    // Set stack size for cuda threads in bytes (size_t)
    // Default is 1024 - If too low, part of scene will render black
    // Needed at high recursion depths
    cudaDeviceSetLimit(cudaLimitStackSize, 4096);
    Color *fb; // Array of RGBA pixels
    size_t fb_size; // WIDTH*HEIGHT*sizeof(Color)
    int pitch = WIDTH*sizeof(Color); // How many bytes to jump to move down a row

    SDL_Window *window;
    SDL_Renderer *renderer;
    SDL_Texture *texture;
    SDL_Event event;

    SDL_Init(SDL_INIT_VIDEO);

    window = SDL_CreateWindow(
        "RayTracer",
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        WIDTH,                             // width, in pixels
        HEIGHT,                            // height, in pixels
        SDL_WINDOW_OPENGL |                // Flags
        //SDL_WINDOW_FULLSCREEN |
        SDL_WINDOW_SHOWN |
        0
    );

    renderer = SDL_CreateRenderer(
        window,
        -1,
        SDL_RENDERER_ACCELERATED // Change to SDL_RENDERER_SOFTWARE if using VNC
    );

    texture = SDL_CreateTexture(
        renderer,
        SDL_PIXELFORMAT_ABGR8888,
        SDL_TEXTUREACCESS_STREAMING,
        WIDTH,
        HEIGHT
    );

    // Adding Spheres to scene
    cudaMallocManaged(&scene.spheres, sizeof(Sphere*)*3);
    AddSphere(1, make_double3(-2.0, 0.0, 4.0), make_uchar4(255, 0, 0, 0), 1, .5);
    AddSphere(1, make_double3(0.0, -1.0, 3.0), make_uchar4(0, 255, 0, 0), 500, .5);
    AddSphere(1, make_double3(2.0, 0.0, 4.0), make_uchar4(0, 0, 255, 0), 500, .5);
    AddSphere(5000, make_double3(0,-5001, 0), make_uchar4(40, 200, 90, 255), 10, .05);
    AddSphere(1, make_double3(0.0, 0.0, -3.0), make_uchar4(0, 0, 0, 0), 0.0, .4);
    AddSphere(2, make_double3(0.0, 1.0, 6.0), make_uchar4(0, 200, 200, 255), 20, 0.5);

    // Adding lights to scene
    cudaMallocManaged(&scene.lights, sizeof(Light*)*3);
    AddLight(AMBIENT, .3);
    AddLight(POINT, 0.6, make_double3(2, 1, 0));
    AddLight(DIRECTIONAL, 0.2, make_double3(1, 4, 4));
    cudaDeviceSynchronize();


    int device = -1;
    cudaGetDevice(&device);
    dim3 blocks(WIDTH/BLOCKX+1,HEIGHT/BLOCKY+1);
    dim3 threads(BLOCKX,BLOCKY);

    fb_size = WIDTH*HEIGHT*sizeof(Color);
    cudaMallocManaged(&fb, fb_size);


    /* Benchmarks */

    // for(int i = 0; i < 1000; i++) {
    //     cudaMemPrefetchAsync(fb, fb_size, device, NULL);
    //     renderSingleFrame<<<blocks,threads>>>(fb, WIDTH, HEIGHT);
    //     cudaDeviceSynchronize();
    //     SDL_UpdateTexture(texture, NULL, (void *)fb, pitch);
    //     SDL_RenderCopy(renderer, texture, NULL, NULL);
    //     SDL_RenderPresent(renderer);
    // }

    // for(int i = 0; i < 100; i++) {
    //     renderSingleFrame<<<blocks,threads>>>(fb, WIDTH, HEIGHT);
    //     cudaDeviceSynchronize();
    //     output_fb_to_sdl(renderer, fb, WIDTH, HEIGHT);
    //     SDL_RenderPresent(renderer);
    // }

    // Render first frame
    renderSingleFrame<<<blocks,threads>>>(fb, WIDTH, HEIGHT);
    cudaDeviceSynchronize();
    SDL_UpdateTexture(texture, NULL, (void *)fb, pitch);
    SDL_RenderCopy(renderer, texture, NULL, NULL);
    SDL_RenderPresent(renderer);

    // Render after every movement keypress
    // wasd,  q-e to rotate, z-x to move up/down
    while (1) {
        SDL_PollEvent(&event);

        if (event.type == SDL_QUIT) {
            break;
        }

        switch(event.type) {
            case SDL_KEYDOWN:
                switch(event.key.keysym.sym) {
                    case SDLK_d:
                        origin.x +=MS*cos(theta);
                        origin.z -=MS*sin(theta);
                        break;
                    case SDLK_a:
                        origin.x -=MS*cos(theta);
                        origin.z +=MS*sin(theta);
                        break;
                    case SDLK_z:
                        origin.y +=MS;
                        break;
                    case SDLK_x:
                        origin.y -=MS;
                        break;
                    case SDLK_s:
                        origin.z -=MS*cos(theta);
                        origin.x -=MS*sin(theta);
                        break;
                    case SDLK_w:
                        origin.z +=MS*cos(theta);
                        origin.x +=MS*sin(theta);
                        break;
                    case SDLK_q:
                        theta -=TR;
                        break;
                    case SDLK_e:
                        theta +=TR;
                        break;
                }

                renderSingleFrame<<<blocks,threads>>>(fb, WIDTH, HEIGHT);
                cudaDeviceSynchronize();
                SDL_UpdateTexture(texture, NULL, (void *)fb, pitch);
                SDL_RenderCopy(renderer, texture, NULL, NULL);
                SDL_RenderPresent(renderer);
                cudaMemPrefetchAsync(fb, fb_size, device, NULL);
                break;
            case SDL_KEYUP:
                break;
            default:
                SDL_RenderPresent(renderer);
                break;
        }
    }

    cudaFree(fb);

    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);

    SDL_Quit();

    return 0;
}

// Old method of screendrawing - slower than using texture
// void output_fb_to_sdl(SDL_Renderer *renderer, Color *framebuffer, int width, int height)
// {
//     for(int i = 0; i < width; i++){
//         for(int j = 0; j < height; j++) {
//            SDL_SetRenderDrawColor(renderer,
//                                   framebuffer[i + j*width].x,
//                                   framebuffer[i + j*width].y,
//                                   framebuffer[i + j*width].z,
//                                   framebuffer[i + j*width].w);
//            SDL_RenderDrawPoint(renderer, i, j);
//         }
//     }
// }

void AddSphere(int radius, double3 center, Color color, double specular, double reflect)
{
    cudaMallocManaged(&scene.spheres[scene.sphere_n], sizeof(Sphere));
    scene.spheres[scene.sphere_n]->radius = radius;
    scene.spheres[scene.sphere_n]->center = center;
    scene.spheres[scene.sphere_n]->color = color;
    scene.spheres[scene.sphere_n]->specular = specular;
    scene.spheres[scene.sphere_n]->reflectivity = reflect;
    scene.sphere_n++;
}

void AddLight(Light_Type type, double intensity)
{
    if(type != AMBIENT) return;
    cudaMallocManaged(&scene.lights[scene.light_n], sizeof(Light));
    scene.lights[scene.light_n]->type = type;
    scene.lights[scene.light_n]->intensity = intensity;
    scene.light_n++;
    return;
}
void AddLight(Light_Type type, double intensity, double3 pos_dir)
{
    if(type == DIRECTIONAL){
        cudaMallocManaged(&scene.lights[scene.light_n], sizeof(Light));
        scene.lights[scene.light_n]->type = type;
        scene.lights[scene.light_n]->intensity = intensity;
        scene.lights[scene.light_n]->dir = pos_dir;
        scene.light_n++;
    } else if(type == POINT){
        cudaMallocManaged(&scene.lights[scene.light_n], sizeof(Light));
        scene.lights[scene.light_n]->type = type;
        scene.lights[scene.light_n]->intensity = intensity;
        scene.lights[scene.light_n]->pos = pos_dir;
        scene.light_n++;
    }
    return;
}

__global__ void renderSingleFrame(Color *framebuffer, int width, int height)
{
    double3 viewport;
    Color color = BG_COLOR;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= width) || (j >= height)) return;

    viewport = CanvasToViewport(i, j, width, height);
    color = TraceRay(origin, viewport, 1.0, DBL_MAX, BG_COLOR, RT_DEPTH);


    framebuffer[i + j*width] = color;
    return;
}

__device__ double3 CanvasToViewport(int x, int y, int width, int height)
{
    double3 retval;
    double3 temp;
    temp.x =  ((double)x-(width/2))/(double)height;
    temp.y = -((double)y-(height/2))/(double)height;
    temp.z = 1.0;

    retval.x = temp.x*cos(theta) + temp.z*sin(theta);
    retval.y = temp.y;
    retval.z = -temp.x*sin(theta) + temp.z*cos(theta);
    return retval;
}

__device__  Color TraceRay(double3 O, double3 viewport, double t_min, double t_max, Color bg_color, int depth)
{
    Color local_color, reflected_color;
    double closest_t;
    Sphere *closest_sphere = NULL;
    int sphere_index = -1;
    closest_t = ClosestIntersection(O, viewport, t_min, t_max, &sphere_index);
    if(sphere_index >= 0) {
        closest_sphere = scene.spheres[sphere_index];
    }

    if (closest_sphere == NULL){
        return bg_color;
    }
    //printf("%lf, %lf\n", t1, t2);
    double3 point = O + closest_t * viewport;
    double3 normal = point - closest_sphere->center;
    normal = (1/length(normal)) * normal;
    local_color =  ComputeLighting(point,
                                   normal,
                                   -1 * viewport,
                                   closest_sphere->specular
                                  )
                   * closest_sphere->color;
    if(depth <= 0 || closest_sphere->reflectivity <= 0) {
        return local_color;
    }

    double3 R = ReflectRay(-1*viewport, normal);
    reflected_color = TraceRay(point, R, 0.001, DBL_MAX, bg_color, depth-1);

    return ((1 - closest_sphere->reflectivity) * local_color)
            + (closest_sphere->reflectivity * reflected_color);
}

__device__ double ClosestIntersection(double3 O, double3 viewport, double t_min, double t_max, int *sphere_index)
{
    double closest_t = DBL_MAX;
    *sphere_index = -1;
    double2 t;

    for (int i = 0; i < scene.sphere_n; i++) {
        t = IntersectRaySphere(O, viewport, scene.spheres[i]);
        if (t.x < closest_t && t.x < t_max && t.x > t_min) {
            closest_t = t.x;
            *sphere_index = i;
        }
        if (t.y < closest_t && t.y < t_max && t.y > t_min) {
            closest_t = t.y;
            *sphere_index = i;
        }
    }

    return closest_t;
}


__device__ double3 ReflectRay(double3 R, double3 normal)
{
     return ((2*dot(normal, R)) * normal) - R;
}
__device__ double2 IntersectRaySphere(double3 O, double3 viewport, Sphere *sphere)
{
    double3 coeffs;
    double discriminant;
    double3 offset = O - sphere->center;

    coeffs.x = dot(viewport, viewport);
    coeffs.y = 2*(dot(offset, viewport));
    coeffs.z = dot(offset, offset) - (sphere->radius * sphere->radius);
    discriminant = (coeffs.y*coeffs.y) - (4*coeffs.x*coeffs.z);

    if(discriminant < 0.0) {
        return make_double2(DBL_MAX, DBL_MAX);
    }

    return make_double2((-coeffs.y + sqrt(discriminant)) / (2*coeffs.x),
                        (-coeffs.y - sqrt(discriminant)) / (2*coeffs.x));
}

__device__ double ComputeLighting(double3 point, double3 normal, double3 view, double spec)
{
    double intensity = 0.0;
    double3 light_vec;
    for(int i = 0; i < scene.light_n; i++) {
        if(scene.lights[i]->type == AMBIENT) {
            intensity += scene.lights[i]->intensity;
        } else {
            if(scene.lights[i]->type == POINT){
                light_vec = scene.lights[i]->pos - point;
            } else {
                light_vec = scene.lights[i]->dir;
            }
            //Shadows
            int shadow_sphere_index;
            ClosestIntersection(point, light_vec, 0.001, DBL_MAX, &shadow_sphere_index);
            //printf("hi\n");
            if(shadow_sphere_index != -1) {
                continue;
            }
            // Diffuse
            double n_dot_l = dot(normal, light_vec);
            if(n_dot_l > 0.0) {
                intensity += scene.lights[i]->intensity*n_dot_l/(length(normal)*length(light_vec));
            }
            // Specular
            if(spec != -1) {
                double3 reflect = ((2*n_dot_l) * normal) - light_vec;
                double r_dot_v = dot(reflect, view);
                if(r_dot_v > 0.0) {
                    intensity += scene.lights[i]->intensity*pow(r_dot_v/(length(reflect)*length(view)), spec);
                }
            }
        }
    }
    return min(intensity, 1.0);
}

__device__ double dot(double3 vec1, double3 vec2)
{
    return vec1.x*vec2.x + vec1.y*vec2.y + vec1.z*vec2.z;
}

__device__ double length(double3 vec)
{
    return sqrt(dot(vec, vec));
}
