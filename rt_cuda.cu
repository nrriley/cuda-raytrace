#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <float.h>
#include <cuda.h>
#include <curand_kernel.h>
#include "vec3.cu"

#define WIDTH 1280
#define HEIGHT 720
#define BLOCKX 16
#define BLOCKY 16
#define BG_COLOR {150, 150, 255, 255}
#define MS .1
#define TR .05
#define RT_DEPTH 5

// Typedefs
typedef uchar4 Color; // .x->R, .y->G, .z->B, .w->A
typedef enum Light_Type {
    AMBIENT,
    DIRECTIONAL,
    POINT
} Light_Type;

typedef struct Material {
    Color color;
    Color emmitance;
    double roughness;
    double reflectance;
    double specular;
} Material;

typedef struct Sphere {
    double3 center;
    double radius;
    Material* material;
} Sphere;



typedef struct Triangle {
    double3 normal;
    double3 v1;
    double3 v2;
    double3 v3;
    Material *material;
} Triangle;

typedef struct Ray {
    double3 origin;
    double3 direction;
} Ray;


typedef struct Interaction {
    double3 point;
    double closest_t;
    double3 normal;
    Material *material;
} Interaction;

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
    Triangle **triangles;
    int sphere_n;
    int triangle_n;
    Light **lights;
    int light_n;
    Material **materials;
    int material_n;
} Scene;

// Host Functions
void output_fb_to_sdl(SDL_Renderer *renderer, Color *framebuffer, int width, int height);
//void AddSphere(int radius, double3 center, Color color, double specular, double reflect);
void AddSphere(int radius, double3 center, Material *material);
Material *AddMaterial(Color color, Color emmitance, double reflectance, double specular, double roughness);
void AddLight(Light_Type type, double intensity);
void AddLight(Light_Type type, double intensity, double3 pos_dir);
// Global Functions
__global__ void setup_curand(curandState *state);
__global__ void renderSingleFrame(Color *framebuffer, int width, int height, curandState *curand_state);

// Device Functions
__device__ double3 CanvasToViewport(int x, int y, int width, int height);
__device__  Color TraceRay(Ray ray, double t_min, double t_max, Color bg_color, int depth);
__device__ double2 IntersectRay(Ray ray, Sphere *sphere);
__device__ double IntersectRay(Ray ray, Triangle *triangle);
__device__ double ComputeLighting(double3 point, double3 normal, double3 view, double spec);
__device__ Interaction ClosestIntersection(Ray ray, double t_min, double t_max);
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

    // // Adding Spheres to scene
    cudaMallocManaged(&scene.materials, sizeof(Material*));
    cudaMallocManaged(&scene.spheres, sizeof(Sphere*));
    Material *material;
    material = AddMaterial(make_uchar4(40, 200, 90, 255), make_uchar4(0,0,0,0), .05, 10, 0);
    AddSphere(5000, make_double3(0,-5001, 0), material);
    material = AddMaterial(make_uchar4(255, 0, 0, 0), make_uchar4(0,0,0,0), .5, 1, 0);
    AddSphere(1, make_double3(-2.0, 0.0, 4.0), material);
    material = AddMaterial(make_uchar4(0, 255, 0, 0), make_uchar4(0,0,0,0), .5, 500, 0);
    AddSphere(1, make_double3(0.0, -1.0, 3.0), material);
    material = AddMaterial(make_uchar4(0, 0, 255, 0), make_uchar4(0,0,0,0), .5, 500, 0);
    AddSphere(1, make_double3(2.0, 0.0, 4.0), material);

    // AddSphere(1, make_double3(-2.0, 0.0, 4.0), make_uchar4(255, 0, 0, 0), 1, .5);
    // AddSphere(1, make_double3(0.0, -1.0, 3.0), make_uchar4(0, 255, 0, 0), 500, .5);
    // AddSphere(1, make_double3(2.0, 0.0, 4.0), make_uchar4(0, 0, 255, 0), 500, .5);
    // AddSphere(5000, make_double3(0,-5001, 0), make_uchar4(40, 200, 90, 255), 10, .05);
    // AddSphere(1, make_double3(0.0, 0.0, -3.0), make_uchar4(0, 0, 0, 0), 0.0, .4);
    // AddSphere(2, make_double3(0.0, 1.0, 6.0), make_uchar4(0, 200, 200, 255), 20, 0.5);

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

    // Setup RNG
    curandState *curand_state;
    cudaMalloc(&curand_state, sizeof(curandState));
    setup_curand<<<blocks,threads>>>(curand_state);


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
    renderSingleFrame<<<blocks,threads>>>(fb, WIDTH, HEIGHT, curand_state);
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

            renderSingleFrame<<<blocks,threads>>>(fb, WIDTH, HEIGHT, curand_state);
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

void AddSphere(int radius, double3 center, Material *material)
{
    cudaMallocManaged(&scene.spheres[scene.sphere_n], sizeof(Sphere));
    scene.spheres[scene.sphere_n]->radius = radius;
    scene.spheres[scene.sphere_n]->center = center;
    scene.spheres[scene.sphere_n]->material = material;
    scene.sphere_n++;
}

Material *AddMaterial(Color color, Color emmitance, double reflectance, double specular, double roughness)
{
    cudaMallocManaged(&scene.materials[scene.material_n], sizeof(Material));
    scene.materials[scene.material_n]->color = color;
    scene.materials[scene.material_n]->emmitance = emmitance;
    scene.materials[scene.material_n]->reflectance = reflectance;
    scene.materials[scene.material_n]->specular = specular;
    scene.materials[scene.material_n]->roughness = roughness;
    scene.material_n++;
    return scene.materials[scene.material_n-1];
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

__global__ void setup_curand(curandState *state)
{
  int idx = threadIdx.x+blockDim.x*blockIdx.x;
  curand_init(1234, idx, 0, &state[idx]);
}

__global__ void renderSingleFrame(Color *framebuffer, int width, int height, curandState *curand_state)
{
    Ray ray;
    ray.origin = origin;
    Color color = BG_COLOR;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= width) || (j >= height)) return;


    ray.direction = CanvasToViewport(i, j, width, height);
    color = TraceRay(ray, 1.0, DBL_MAX, BG_COLOR, RT_DEPTH);
    curandState state = curand_state[i];
    printf("%f\t", curand_uniform(&(state)));

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

__device__  Color TraceRay(Ray ray, double t_min, double t_max, Color bg_color, int depth)
{
    Color local_color, reflected_color;
    Ray reflected_ray;
    Interaction interaction;



    interaction = ClosestIntersection(ray, t_min, t_max);

    if (interaction.closest_t == DBL_MAX){
        return bg_color;
    }
    double3 point = ray.origin + interaction.closest_t * ray.direction;
    double3 normal = interaction.normal;
    normal = (1/length(normal)) * normal;
    local_color =  ComputeLighting(point,
                                   normal,
                                   -1 * ray.direction,
                                   interaction.material->specular
                                  )
                                  * interaction.material->color;
    if(depth <= 0 || interaction.material->reflectance <= 0) {
        return local_color;
    }

    reflected_ray.origin = point;
    reflected_ray.direction = ReflectRay(-1*ray.direction, normal);
    reflected_color = TraceRay(reflected_ray, .001, DBL_MAX, bg_color, depth-1);

    return ((1 - interaction.material->reflectance) * local_color)
            + (interaction.material->reflectance * reflected_color);
}

__device__ Interaction ClosestIntersection(Ray ray, double t_min, double t_max)
{
    Interaction interaction;
    double2 t;

    interaction.closest_t = DBL_MAX;

    for (int i = 0; i < scene.sphere_n; i++) {
        t = IntersectRay(ray, scene.spheres[i]);
        if (t.x < interaction.closest_t && t.x < t_max && t.x > t_min) {
            interaction.closest_t = t.x;
            interaction.point = ray.origin + interaction.closest_t * ray.direction;
            interaction.normal = interaction.point - scene.spheres[i]->center;
            interaction.material = scene.spheres[i]->material;
        }
        if (t.y < interaction.closest_t && t.y < t_max && t.y > t_min) {
            interaction.closest_t = t.y;
            interaction.point = ray.origin + interaction.closest_t * ray.direction;
            interaction.normal = interaction.point - scene.spheres[i]->center;
            interaction.material = scene.spheres[i]->material;
        }
    }

    for (int i = 0; i < scene.triangle_n; i++) {
        t.x = IntersectRay(ray, scene.triangles[i]);
        if (t.x < interaction.closest_t && t.x < t_max && t.x > t_min) {
            interaction.closest_t = t.x;
            interaction.point = ray.origin + interaction.closest_t * ray.direction;
            interaction.normal = scene.triangles[i]->normal;
            interaction.material = scene.triangles[i]->material;
        }
    }

    return interaction;
}


__device__ double3 ReflectRay(double3 R, double3 normal)
{
    return ((2*dot(normal, R)) * normal) - R;
}
__device__ double2 IntersectRay(Ray ray, Sphere *sphere)
{
    double3 coeffs;
    double discriminant;
    double3 offset = ray.origin - sphere->center;

    coeffs.x = dot(ray.direction, ray.direction);
    coeffs.y = 2*(dot(offset, ray.direction));
    coeffs.z = dot(offset, offset) - (sphere->radius * sphere->radius);
    discriminant = (coeffs.y*coeffs.y) - (4*coeffs.x*coeffs.z);

    if(discriminant < 0.0) {
        return make_double2(DBL_MAX, DBL_MAX);
    }

    return make_double2((-coeffs.y + sqrt(discriminant)) / (2*coeffs.x),
    (-coeffs.y - sqrt(discriminant)) / (2*coeffs.x));
}

__device__ double IntersectRay(Ray ray, Triangle *triangle)
{
    //TODO
    return 0.0;
}

__device__ double ComputeLighting(double3 point, double3 normal, double3 view, double spec)
{
    double intensity = 0.0;
    Ray light_ray;
    light_ray.origin = point;
    for(int i = 0; i < scene.light_n; i++) {
        if(scene.lights[i]->type == AMBIENT) {
            intensity += scene.lights[i]->intensity;
        } else {
            if(scene.lights[i]->type == POINT){
                light_ray.direction = scene.lights[i]->pos - point;
            } else {
                light_ray.direction = scene.lights[i]->dir;
            }
            //Shadows
            if(ClosestIntersection(light_ray, 0.001, DBL_MAX).closest_t < DBL_MAX) {
                // If object is occluding light then go to next light
                continue;
            }
            // Diffuse
            double n_dot_l = dot(normal, light_ray.direction);
            if(n_dot_l > 0.0) {
                intensity += scene.lights[i]->intensity*n_dot_l/(length(normal)*length(light_ray.direction));
            }
            //Specular
            if(spec != -1) {
                double3 reflect = ((2*n_dot_l) * normal) - light_ray.direction;
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
