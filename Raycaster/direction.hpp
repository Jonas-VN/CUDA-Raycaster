#pragma once
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

struct Direction {
    double x;
    double y;

    __host__ __device__ Direction(double x, double y) : x(x), y(y) {}

    __host__ __device__ Direction() : x(0), y(0) {}

    __host__ __device__ void dotProduct(const double matrix[2][2]) {
        double x_ = x;
        double y_ = y;
        x = matrix[0][0] * x_ + matrix[0][1] * y_;
        y = matrix[1][0] * x_ + matrix[1][1] * y_;
    }

    __host__ __device__ double dotProduct(const Direction* other) const {
        return x * other->x + y * other->y;
    }

    __host__ __device__ double dotProduct(const Direction other) const {
        return x * other.x + y * other.y;
    }

    __host__ __device__ void normalize() {
        double magnitude = sqrt(x * x + y * y);
        if (magnitude != 0.0) {
            x /= magnitude;
            y /= magnitude;
        }
    }
};
