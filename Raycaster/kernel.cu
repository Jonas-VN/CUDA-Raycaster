#include <SDL.h>
#include <stdio.h>
#include "settings.hpp"
#include "player.h"
#include "map.hpp"
#include <iostream>
#include <string>


SDL_Window* gWindow = nullptr;
SDL_Surface* gSurface = nullptr;

bool initSDL() {
    // Initialize SDL
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        printf("SDL could not initialize! SDL_Error: %s\n", SDL_GetError());
        return false;
    }

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

    // Quit SDL subsystems
    SDL_Quit();
}

void CPU_Raycast(Uint32* pixels, Player* player, const Map* map) {
    // Clear the screen (set every pixel to black)
    memset(pixels, SDL_MapRGB(gSurface->format, 0, 0, 0), SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(Uint32));

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

        Coordinate intersection;
        int hitDirection = 0;
        double distanceToWall = 0.0;
        bool hit = false;
        while (!hit) {
            if (verticalDistance + y * delta_v < horizontalDistance + x * delta_h) {
                factor = verticalDistance + y * delta_v;
                if (rayDirection.x < 0) {
                    intersection = Coordinate(
                        std::round(rayDirection.x * factor + player->coordinate->x) - 1.0,
                        std::floor(rayDirection.y * factor + player->coordinate->y)
                    );
                }
                else {
                    intersection = Coordinate(
                        std::round(rayDirection.x * factor + player->coordinate->x),
                        std::floor(rayDirection.y * factor + player->coordinate->y)
                    );
                }
                if (map->isWall(&intersection)) {
                    hit = true;
                    distanceToWall = factor * rayDirection.dotProduct(player->direction);
                    hitDirection = 1;
                }
                y++;
            }
            else {
                factor = horizontalDistance + x * delta_h;
                if (rayDirection.y < 0) {
                    intersection = Coordinate(
                        std::floor(rayDirection.x * factor + player->coordinate->x),
                        std::round(rayDirection.y * factor + player->coordinate->y) - 1.0
                    );
                }
                else {
                    intersection = Coordinate(
                        std::floor(rayDirection.x * factor + player->coordinate->x),
                        std::round(rayDirection.y * factor + player->coordinate->y)
                    );
                }

                if (map->isWall(&intersection)) {
                    hit = true;
                    distanceToWall = factor * rayDirection.dotProduct(player->direction);
                }
                x++;
            }
        }

        double length = 1 / distanceToWall * SCREEN_HEIGHT;
        int start = (SCREEN_HEIGHT - length) / 2 >= 0 ? (int)(SCREEN_HEIGHT - length) / 2 : 0;
        int end = start + length <= SCREEN_HEIGHT ? (int)start + length : SCREEN_HEIGHT;

        for (int y = start; y < end; ++y) {
            pixels[y * SCREEN_WIDTH + column] = SDL_MapRGB(gSurface->format, 255 - hitDirection * 100, 0, 0);
        }
    }
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

    double currentTime = SDL_GetTicks64();
    double prevTime = currentTime;
    bool quit = false;
    while (!quit) {
        currentTime = SDL_GetTicks64();
        double delta = (currentTime - prevTime) / 1000;
        prevTime = currentTime;
        int fps = (int) 1 / delta;

        if (SDL_LockSurface(gSurface) == 0) {
            Uint32* pixels = (Uint32*)gSurface->pixels;
            CPU_Raycast(pixels, player, map);
            gSurface->pixels = pixels;
            SDL_UnlockSurface(gSurface);
            SDL_UpdateWindowSurface(gWindow);
            std::string windowTitle = "Raycaster (FPS: " + std::to_string(fps) + ")";
            SDL_SetWindowTitle(gWindow, windowTitle.c_str());
            quit = handle_keys(delta, player);
        }
    }

    closeSDL();

    delete(player);
    delete(map);

    return 0;
}



