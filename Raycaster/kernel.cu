﻿#include <SDL.h>
#include <stdio.h>
#include "settings.hpp"
#include "player.h"
#include "map.hpp"
#include <iostream>
#include <string>

#if USE_GPU
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#endif

#if USE_TEXTURE
#include <SDL_image.h>

Uint32* gTexture = nullptr;
#endif


SDL_Window* gWindow = nullptr;
SDL_Surface* gSurface = nullptr;
int frame = 0;
bool test = false;


#if USE_GPU
__global__ void GPU_Raycast(Uint32* pixels, double playerDirectionX, double playerDirectionY, double playerCoordinateX, double playerCoordinateY, double cameraDirectionX, double cameraDirectionY, double cameraDistance) {
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column < SCREEN_WIDTH) {
        // Calculate ray direction
        double factor = -1.0 + 2.0 * column / SCREEN_WIDTH;
        double rayDirectionX = cameraDistance * playerDirectionX + factor * cameraDirectionX;
        double rayDirectionY = cameraDistance * playerDirectionY + factor * cameraDirectionY;
        double magnitude = sqrt(rayDirectionX * rayDirectionX + rayDirectionY * rayDirectionY);
        if (magnitude != 0) {
            rayDirectionX /= magnitude;
            rayDirectionY /= magnitude;
        }

        // Raycast
        int x = 0;
        int y = 0;
        double delta_v = abs(1 / rayDirectionX);
        double delta_h = abs(1 / rayDirectionY);

        double verticalDistance;
        if (rayDirectionX < 0) verticalDistance = (playerCoordinateX - floor(playerCoordinateX)) * delta_v;
        else verticalDistance = (ceil(playerCoordinateX) - playerCoordinateX) * delta_v;

        double horizontalDistance;
        if (rayDirectionY < 0) horizontalDistance = (playerCoordinateY - floor(playerCoordinateY)) * delta_h;
        else horizontalDistance = (ceil(playerCoordinateY) - playerCoordinateY) * delta_h;

        double intersectionX;
        double intersectionY;
        int hitDirection = 0;
        double distanceToWall = 0.0;
        bool hit = false;
        while (!hit) {
            if (verticalDistance + y * delta_v < horizontalDistance + x * delta_h) {
                factor = verticalDistance + y * delta_v;
                if (rayDirectionX < 0) {
                    intersectionX = round(rayDirectionX * factor + playerCoordinateX) - 1.0;
                    intersectionY = floor(rayDirectionY * factor + playerCoordinateY);
                }
                else {
                    intersectionX = round(rayDirectionX * factor + playerCoordinateX);
                    intersectionY = floor(rayDirectionY * factor + playerCoordinateY);
                }
                if (d_map[(int) intersectionY][(int) intersectionX] == 1) {
                    hit = true;
                    distanceToWall = factor * (rayDirectionX * playerDirectionX + rayDirectionY * playerDirectionY);
                    hitDirection = 1;
                }
                y++;
            }
            else {
                factor = horizontalDistance + x * delta_h;
                if (rayDirectionY < 0) {
                    intersectionX = floor(rayDirectionX * factor + playerCoordinateX);
                    intersectionY = round(rayDirectionY * factor + playerCoordinateY) - 1.0;
                }
                else {
                    intersectionX = floor(rayDirectionX * factor + playerCoordinateX);
                    intersectionY = round(rayDirectionY * factor + playerCoordinateY);
                }

                if (d_map[(int)intersectionY][(int)intersectionX] == 1) {
                    hit = true;
                    distanceToWall = factor * (rayDirectionX * playerDirectionX + rayDirectionY * playerDirectionY);
                }
                x++;
            }
        }

        double length = 1 / distanceToWall * SCREEN_HEIGHT;
        int start = (SCREEN_HEIGHT - length) / 2 >= 0 ? (int)(SCREEN_HEIGHT - length) / 2 : 0;
        int end = start + length <= SCREEN_HEIGHT ? (int)start + length : SCREEN_HEIGHT;
        for (int y = start; y < end; ++y) {
            pixels[y * SCREEN_WIDTH + column] = hitDirection == 0 ? 0xFF0000 : 0xDD0000;
        }
    }
}
#else
void CPU_Raycast(Uint32* pixels, Player* player, const Map* map) {
    double factor;
    for (int column = 0; column < SCREEN_WIDTH; column++) {

        // Calculate ray direction
        factor = -1.0 + 2.0 * column / SCREEN_WIDTH;
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
        if (rayDirection.x < 0) verticalDistance = (player->coordinate->x - std::floor(player->coordinate->x)) * delta_v;
        else verticalDistance = (std::ceil(player->coordinate->x) - player->coordinate->x) * delta_v;

        double horizontalDistance;
        if (rayDirection.y < 0) horizontalDistance = (player->coordinate->y - std::floor(player->coordinate->y)) * delta_h;
        else horizontalDistance = (std::ceil(player->coordinate->y) - player->coordinate->y) * delta_h;

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
                        std::round(intersection.x) - 1.0,
                        std::floor(intersection.y)
                    );
                }
                else {
                    roundedIntersection = Coordinate(
                        std::round(intersection.x),
                        std::floor(intersection.y)
                    );
                }
                if (map->isWall(&roundedIntersection)) {
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
                        std::floor(intersection.x),
                        std::round(intersection.y) - 1.0
                    );
                }
                else {
                    roundedIntersection = Coordinate(
                        std::floor(intersection.x),
                        std::round(intersection.y)
                    );
                }

                if (map->isWall(&roundedIntersection)) {
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
        if (hitDirection == 0) textureX = (int) ((intersection.x - std::floor(intersection.x)) * TEXTURE_WIDTH);
        else textureX = (int) ((intersection.y - std::floor(intersection.y)) * TEXTURE_WIDTH);
        double ratio = (double) TEXTURE_HEIGHT / (end - start);
        double textureY = start >= 0 ? 0.0 : -start;

        for (int y = realStart; y < realEnd; ++y) {
            int sourceY = (int) (textureY * ratio);
            pixels[y * SCREEN_WIDTH + column] = gTexture[sourceY * TEXTURE_WIDTH + textureX];
            textureY++;
        }

#else
        for (int y = realStart; y < realEnd; ++y) {
            pixels[y * SCREEN_WIDTH + column] = SDL_MapRGB(gSurface->format, 255 - hitDirection * 100, 0, 0);
        }
#endif
    }
}
#endif


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

    // Load image
    SDL_Surface* surface = IMG_Load("src/wall.png");
    surface = SDL_ConvertSurfaceFormat(surface, SDL_PIXELFORMAT_RGBA32, 0);
    if (!surface) {
        std::cerr << "Failed to load image: " << IMG_GetError() << std::endl;
        IMG_Quit();
        SDL_Quit();
        return -1;
    }
    gTexture = static_cast<Uint32*>(surface->pixels);
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

bool handle_keys(double delta, Player* player) {
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

#if USE_GPU
    map->copyMapToGPU();
    Uint32* gpuPixels;
    cudaMalloc((void**)&gpuPixels, SCREEN_HEIGHT * SCREEN_WIDTH * sizeof(Uint32));
    int blockSize = 256;
    int numBlocks = (SCREEN_WIDTH + blockSize - 1) / blockSize;
    std::cout << numBlocks << std::endl;
#endif

    double currentTime = (double) SDL_GetTicks64();
    double prevTime = currentTime;
    bool quit = false;
    while (!quit) {
        currentTime = (double) SDL_GetTicks64();
        double delta = (currentTime - prevTime) / 1000.0;
        prevTime = currentTime;
        int fps = (int) (1 / delta);

        if (SDL_LockSurface(gSurface) == 0) {
            if (frame++ % 200 == 0) test = !test;
            Uint32* pixels = (Uint32*)gSurface->pixels;

#if USE_GPU
            cudaMemset(gpuPixels, 0x000000, SCREEN_HEIGHT * SCREEN_WIDTH * sizeof(Uint32));
            GPU_Raycast << <numBlocks, blockSize >> > (gpuPixels, player->direction->x, player->direction->y, player->coordinate->x, player->coordinate->y, player->camera->direction->x, player->camera->direction->y, player->camera->distanceToPlayer);
            quit = handle_keys(delta, player);
            cudaDeviceSynchronize();
            cudaMemcpy(pixels, gpuPixels, SCREEN_HEIGHT * SCREEN_WIDTH * sizeof(Uint32), cudaMemcpyDeviceToHost);
#else
            memset(pixels, 0x000000, SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(Uint32));
            CPU_Raycast(pixels, player, map);
            double ratio = 0.4;
            int newWidth = TEXTURE_WIDTH / ratio;
            int newHeight = TEXTURE_HEIGHT / ratio;

            for (int y = 0; y < newHeight; ++y) {
                for (int x = 0; x < newWidth; ++x) {
                    int sourceX = (int) (x * ratio);
                    int sourceY = (int) (y * ratio);

                    // Copy pixel value from original image to resized image
                    //pixels[y * SCREEN_WIDTH + x] = gTexture[sourceY * TEXTURE_WIDTH + sourceX];
                }
            }


            quit = handle_keys(delta, player);
#endif

            gSurface->pixels = pixels;
            SDL_UnlockSurface(gSurface);
            SDL_UpdateWindowSurface(gWindow);
            std::string windowTitle = "Raycaster (FPS: " + std::to_string(fps) + ")";
            SDL_SetWindowTitle(gWindow, windowTitle.c_str());
        }
    }

    closeSDL();

    delete(player);
    delete(map);

#if USE_GPU
    cudaFree(gpuPixels);
#endif
    return 0;
}



