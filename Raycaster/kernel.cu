#include <SDL.h>
#include <stdio.h>
#include "settings.hpp"
#include "player.h"
#include "map.hpp"
#include <iostream>
#include <string>

#if TEST_MEMORY_LEAKS
#include <crtdbg.h>
#endif

#if USE_GPU
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__constant__ int gpuMap[MAP_HEIGHT][MAP_WIDTH];
#endif

#if USE_TEXTURE
#include <SDL_image.h>
#endif

SDL_Window* gWindow = nullptr;
SDL_Surface* gSurface = nullptr;

// This may be ugly code, but it does show the difference between GPU and CPU code rather easily without much effort
#if USE_GPU && USE_TEXTURE && USE_TEXTURE_OBJECT
__global__ void raycast(Uint32* pixels, Player* player, cudaTextureObject_t texture) {
#elif USE_GPU && USE_TEXTURE && !USE_TEXTURE_OBJECT
__global__ void raycast(Uint32 * pixels, Player * player, Uint32* texture) {
#elif USE_GPU && !USE_TEXTURE
__global__ void raycast(Uint32 * pixels, Player * player) {
#elif !USE_GPU && USE_TEXTURE
void raycast(Uint32 * pixels, Player * player, const Map * map, Uint32 * texture) {
#elif !USE_GPU && !USE_TEXTURE
void raycast(Uint32 * pixels, Player * player, const Map * map) {
#endif

// On the GPU: if statment to check bounds
// On the CPU: for loop
#if USE_GPU
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column < SCREEN_WIDTH) {
#else
    for (int column = 0; column < SCREEN_WIDTH; column++) {
#endif
        // Calculate ray direction
        double factor = -1.0 + 2.0 * column / SCREEN_WIDTH;
        Direction rayDirection = Direction(
            player->camera->distanceToPlayer * player->direction->x + factor * player->camera->direction->x,
            player->camera->distanceToPlayer * player->direction->y + factor * player->camera->direction->y
        );
        rayDirection.normalize();

        // Raycast
        int x = 0;
        int y = 0;
        double delta_v = abs(1 / rayDirection.x);
        double delta_h = abs(1 / rayDirection.y);

        double verticalDistance;
        if (rayDirection.x < 0) verticalDistance = (player->coordinate->x - floor(player->coordinate->x)) * delta_v;
        else verticalDistance = (ceil(player->coordinate->x) - player->coordinate->x) * delta_v;

        double horizontalDistance;
        if (rayDirection.y < 0) horizontalDistance = (player->coordinate->y - floor(player->coordinate->y)) * delta_h;
        else horizontalDistance = (ceil(player->coordinate->y) - player->coordinate->y) * delta_h;

        Coordinate roundedIntersection;
        Coordinate intersection;
        int hitDirection = 0;
        double distanceToWall = 0.0;
        bool hit = false;
        while (!hit) {
            if (verticalDistance + y * delta_v < horizontalDistance + x * delta_h) {
                factor = verticalDistance + y * delta_v;
                intersection = Coordinate(
                    rayDirection.x * factor + player->coordinate->x,
                    rayDirection.y * factor + player->coordinate->y
                );

                if (rayDirection.x < 0) {
                    roundedIntersection = Coordinate(
                        round(intersection.x) - 1.0,
                        floor(intersection.y)
                    );
                }
                else {
                    roundedIntersection = Coordinate(
                        round(intersection.x),
                        floor(intersection.y)
                    );
                }
#if USE_GPU
                if (gpuMap[(int)roundedIntersection.y][(int)roundedIntersection.x]) {
#else
                if (map->isWall(&roundedIntersection)) {
#endif
                    hit = true;
                    distanceToWall = factor * rayDirection.dotProduct(player->direction);
                    hitDirection = 1;
                }
                y++;
                }
            else {
                factor = horizontalDistance + x * delta_h;
                intersection = Coordinate(
                    rayDirection.x * factor + player->coordinate->x,
                    rayDirection.y * factor + player->coordinate->y
                );
                if (rayDirection.y < 0) {
                    roundedIntersection = Coordinate(
                        floor(intersection.x),
                        round(intersection.y) - 1.0
                    );
                }
                else {
                    roundedIntersection = Coordinate(
                        floor(intersection.x),
                        round(intersection.y)
                    );
                }
#if USE_GPU
                if (gpuMap[(int)roundedIntersection.y][(int)roundedIntersection.x]) {
#else
                if (map->isWall(&roundedIntersection)) {
#endif
                    hit = true;
                    distanceToWall = factor * rayDirection.dotProduct(player->direction);
                }
                x++;
                }
            }

        double length = 1 / distanceToWall * SCREEN_HEIGHT;
        int start = (SCREEN_HEIGHT - length) / 2;
        int end = start + length;

        int realStart = start >= 0 ? start : 0;
        int realEnd = end <= SCREEN_HEIGHT ? end : SCREEN_HEIGHT;

#if USE_TEXTURE
        int textureX;
        if (hitDirection == 0) textureX = (int)((intersection.x - floor(intersection.x)) * TEXTURE_WIDTH);
        else textureX = (int)((intersection.y - floor(intersection.y)) * TEXTURE_WIDTH);
        double ratio = (double)TEXTURE_HEIGHT / (end - start);
        double textureY = start >= 0 ? 0.0 : -start;
#endif

        for (int y = realStart; y < realEnd; ++y) {
#if USE_TEXTURE
            int sourceY = (int)(textureY++ * ratio);
#endif
#if USE_GPU && USE_TEXTURE && USE_TEXTURE_OBJECT
            pixels[y * SCREEN_WIDTH + column] = tex1Dfetch<Uint32>(texture, sourceY * TEXTURE_WIDTH + textureX);
#elif USE_GPU && USE_TEXTURE && !USE_TEXTURE_OBJECT
            pixels[y * SCREEN_WIDTH + column] = texture[sourceY * TEXTURE_WIDTH + textureX];
#elif !USE_GPU && USE_TEXTURE
            pixels[y * SCREEN_WIDTH + column] = texture[sourceY * TEXTURE_WIDTH + textureX];
#else
            pixels[y * SCREEN_WIDTH + column] = 0xFF0000 - hitDirection * 0x330000;
#endif
        }
    }
}

bool initSDL() {
    // Initialize SDL
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        printf("SDL could not initialize! SDL_Error: %s\n", SDL_GetError());
        return false;
    }

#if USE_TEXTURE
    // Initialize SDL_image
    int imgFlags = IMG_INIT_PNG;
    if ((IMG_Init(imgFlags) & imgFlags) != imgFlags) {
        std::cerr << "SDL_image initialization failed: " << IMG_GetError() << std::endl;
        SDL_Quit();
        return false;
    }
#endif

    // Create window
    gWindow = SDL_CreateWindow("Raycaster (FPS: )", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_SHOWN);
    if (gWindow == nullptr) {
        printf("Window could not be created! SDL_Error: %s\n", SDL_GetError());
        return false;
    }

    // Get window surface
    gSurface = SDL_GetWindowSurface(gWindow);
    SDL_SetRelativeMouseMode(SDL_TRUE);

    return true;
}

void closeSDL() {
    // Destroy window
    SDL_DestroyWindow(gWindow);
    gWindow = nullptr;

#if USE_TEXTURE
    // Quit SDL_image
    IMG_Quit();
#endif

    // Quit SDL subsystems
    SDL_Quit();
}

#if USE_TEXTURE
Uint32* loadImage(const char* filepath) {
    SDL_Surface* surface = IMG_Load(filepath);
    surface = SDL_ConvertSurfaceFormat(surface, SDL_PIXELFORMAT_RGBA32, 0);
    if (!surface) {
        std::cerr << "Failed to load image: " << IMG_GetError() << std::endl;
        IMG_Quit();
        SDL_Quit();
        return false;
    }
    return static_cast<Uint32*>(surface->pixels);
}
#endif

bool handle_keys(double delta, Player * player) {
    SDL_Event e;
    bool quit = false;

    while (SDL_PollEvent(&e) != 0) {
        if (e.type == SDL_QUIT) {
            quit = true;
        }
        else if (e.type == SDL_MOUSEMOTION) {
            player->rotate(e.motion.xrel);
        }
    }

    const Uint8* keys = SDL_GetKeyboardState(NULL);
    if (keys[SDL_SCANCODE_ESCAPE]) {
        quit = true;
    }
    if (keys[SDL_SCANCODE_W]) {
        player->moveY(delta);
    }
    if (keys[SDL_SCANCODE_S]) {
        player->moveY(-delta);
    }
    if (keys[SDL_SCANCODE_D]) {
        player->moveX(delta);
    }
    if (keys[SDL_SCANCODE_A]) {
        player->moveX(-delta);
    }

    return quit;
}


int main(int argc, char* args[]) {
    // Initialize SDL
    if (!initSDL()) {
        printf("Failed to initialize!\n");
        return -1;
    }

    const Map* map = new Map();
    Player* player = new Player(map);

    double totalDeltaTime = 0.0;
    int numFrames = 0;

#if USE_TEXTURE
    Uint32* texture = loadImage("src/wall.png");
#endif

#if USE_GPU
    cudaMemcpyToSymbol(gpuMap, map, MAP_HEIGHT * MAP_WIDTH * sizeof(int));

    Uint32* gpuPixels;
    Player* gpuPlayer;
    Camera* gpuCamera;
    Direction* gpuPlayerDirection;
    Coordinate* gpuPlayerCoordinate;
    Direction* gpuCameraDirection;
    cudaMalloc((void**)&gpuPixels, SCREEN_HEIGHT * SCREEN_WIDTH * sizeof(Uint32));
    cudaMalloc((void**)&gpuPlayer, sizeof(Player));
    cudaMalloc((void**)&gpuCamera, sizeof(Camera));
    cudaMalloc((void**)&gpuPlayerDirection, sizeof(Direction));
    cudaMalloc((void**)&gpuPlayerCoordinate, sizeof(Coordinate));
    cudaMalloc((void**)&gpuCameraDirection, sizeof(Direction));

    // The player & camera are technically constant since only the pointers are relevant or the other fields are constant
    cudaMemcpy(gpuPlayer, player, sizeof(Player), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuCamera, player->camera, sizeof(Camera), cudaMemcpyHostToDevice);

    // Populate pointers (these can be copied in advance because the gpuPlayer & gpuCamera never changes and thus the pointers stay relevant)
    cudaMemcpy(&(gpuPlayer->direction), &gpuPlayerDirection, sizeof(Direction*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(gpuPlayer->coordinate), &gpuPlayerCoordinate, sizeof(Coordinate*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(gpuPlayer->camera), &gpuCamera, sizeof(Camera*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(gpuCamera->direction), &gpuCameraDirection, sizeof(Direction*), cudaMemcpyHostToDevice);

    int blockSize = 128;
    int numBlocks = (SCREEN_WIDTH + blockSize - 1) / blockSize;
#endif

#if USE_GPU && USE_TEXTURE
    Uint32* textureData;
    cudaMalloc((void**)&textureData, TEXTURE_HEIGHT * TEXTURE_WIDTH * sizeof(Uint32));
    cudaMemcpy(textureData, texture, TEXTURE_HEIGHT * TEXTURE_WIDTH * sizeof(Uint32), cudaMemcpyHostToDevice);
#endif

#if USE_GPU && USE_TEXTURE && USE_TEXTURE_OBJECT
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = textureData;
    resDesc.res.linear.desc.x = 32; // bits per channel
    resDesc.res.linear.sizeInBytes = TEXTURE_HEIGHT * TEXTURE_WIDTH * sizeof(Uint32);

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;

    cudaTextureObject_t gpuTexture = 0;
    cudaCreateTextureObject(&gpuTexture, &resDesc, &texDesc, NULL);
#endif

    double currentTime = (double)SDL_GetTicks64();
    double prevTime = currentTime;
    bool quit = false;
    while (!quit) {
        currentTime = (double)SDL_GetTicks64();
        double delta = (currentTime - prevTime) / 1000.0;
        totalDeltaTime += delta;
        numFrames++;
        prevTime = currentTime;
        int fps = (int)(1 / delta);

        if (SDL_LockSurface(gSurface) == 0) {
            Uint32* pixels = (Uint32*)gSurface->pixels;

#if USE_GPU
            cudaMemset(gpuPixels, BACKGROUND_COLOR, SCREEN_HEIGHT * SCREEN_WIDTH * sizeof(Uint32));
            // Copy the content of the player
            cudaMemcpy(gpuPlayerDirection, player->direction, sizeof(Direction), cudaMemcpyHostToDevice);
            cudaMemcpy(gpuPlayerCoordinate, player->coordinate, sizeof(Coordinate), cudaMemcpyHostToDevice);
            cudaMemcpy(gpuCameraDirection, player->camera->direction, sizeof(Direction), cudaMemcpyHostToDevice);
#else
            memset(pixels, BACKGROUND_COLOR, SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(Uint32));
#endif

#if USE_GPU && USE_TEXTURE && USE_TEXTURE_OBJECT
            raycast << <numBlocks, blockSize >> > (gpuPixels, gpuPlayer, gpuTexture);
#elif USE_GPU && USE_TEXTURE && !USE_TEXTURE_OBJECT
            raycast << <numBlocks, blockSize >> > (gpuPixels, gpuPlayer, textureData);
#elif USE_GPU && !USE_TEXTURE
            raycast << <numBlocks, blockSize >> > (gpuPixels, gpuPlayer);
#elif !USE_GPU && USE_TEXTURE
            raycast(pixels, player, map, texture);
#elif !USE_GPU && !USE_TEXTURE
            raycast(pixels, player, map);
#endif

            quit = handle_keys(delta, player);

#if USE_GPU
            cudaDeviceSynchronize();
            cudaMemcpy(pixels, gpuPixels, SCREEN_HEIGHT * SCREEN_WIDTH * sizeof(Uint32), cudaMemcpyDeviceToHost);
#endif

            gSurface->pixels = pixels;
            SDL_UnlockSurface(gSurface);
            SDL_UpdateWindowSurface(gWindow);
            std::string windowTitle = "Raycaster (FPS: " + std::to_string(fps) + ")";
            SDL_SetWindowTitle(gWindow, windowTitle.c_str());

#if BENCHMARKING
            if (totalDeltaTime >= 10) quit = true;
#endif
        }
    }

    closeSDL();

    delete(player);
    delete(map);

#if USE_GPU
    cudaFree(gpuPixels);
    cudaFree(gpuPlayer);
    cudaFree(gpuPlayerDirection);
    cudaFree(gpuPlayerCoordinate);
    cudaFree(gpuCameraDirection);
    cudaFree(gpuCamera);
#endif

#if USE_GPU && USE_TEXTURE && USE_TEXTURE_OBJECT
    cudaDestroyTextureObject(gpuTexture);
#endif

#if USE_GPU && USE_TEXTURE
    cudaFree(textureData);
#endif

    std::cout << "Average FPS: " << 1 / (totalDeltaTime / numFrames) << std::endl;

#if TEST_MEMORY_LEAKS
    _CrtDumpMemoryLeaks();
#endif

    return 0;
}