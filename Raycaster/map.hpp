#pragma once
#include "coordinate.hpp"
#include "settings.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#if USE_GPU
__constant__ int d_map[MAP_HEIGHT][MAP_WIDTH];
#endif

class Map {
public:
    int map[MAP_HEIGHT][MAP_WIDTH] = {
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
        {1, 0, 0, 0, 0, 0, 0, 0, 0, 1},
        {1, 1, 0, 0, 1, 0, 0, 0, 0, 1},
        {1, 0, 0, 0, 0, 0, 0, 1, 0, 1},
        {1, 1, 0, 0, 1, 0, 0, 1, 0, 1},
        {1, 0, 0, 0, 0, 0, 0, 1, 0, 1},
        {1, 0, 0, 0, 0, 0, 0, 1, 0, 1},
        {1, 0, 0, 0, 0, 0, 0, 0, 0, 1},
        {1, 0, 0, 0, 0, 0, 0, 0, 0, 1},
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
    };

public:
    bool isWall(Coordinate* coordinate) const {
        return map[(int)(coordinate->y)][(int)(coordinate->x)] == 1;
    }

#if USE_GPU
    void copyMapToGPU() const {
        cudaMemcpyToSymbol(d_map, map, MAP_HEIGHT * MAP_WIDTH * sizeof(int));
    }
#endif
};
