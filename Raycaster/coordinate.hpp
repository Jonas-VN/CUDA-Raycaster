#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


struct Coordinate {
    double x;
    double y;

    __host__ __device__ Coordinate(double x, double y) : x(x), y(y) {}

    __host__ __device__ Coordinate() : x(0), y(0) {}
};