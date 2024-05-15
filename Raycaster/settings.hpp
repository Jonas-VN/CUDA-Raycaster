#pragma once
#include <cmath>

#define USE_GPU true
#define USE_TEXTURE true
#define USE_TEXTURE_OBJECT true

#define SCREEN_WIDTH 1900
#define SCREEN_HEIGHT 1000

#define TEXTURE_WIDTH 360
#define TEXTURE_HEIGHT 360

#define DEG2RAD 3.14159265358979323846 / 180
#define POV 90

#define MAP_WIDTH 10
#define MAP_HEIGHT 10

#define PLAYER_SPEED 1
#define PLAYER_SENSITIVITY 1 / 75

#define PLAYER_START_COORDINATE_X 3 + 1 / sqrt(2)
#define PLAYER_START_COORDINATE_Y 4 - 1 / sqrt(2)
#define PLAYER_START_DIRECTION_X 1 / sqrt(2)
#define PLAYER_START_DIRECTION_Y -1 / sqrt(2)

#define BACKGROUND_COLOR 0x000000

#define TEST_MEMORY_LEAKS false

#define BENCHMARKING false