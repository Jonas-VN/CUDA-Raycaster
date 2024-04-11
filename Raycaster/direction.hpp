#pragma once
#include <cmath>

struct Direction {
    double x;
    double y;

    Direction(double x, double y) : x(x), y(y) {}

    void dotProduct(const double matrix[2][2]) {
        double x_ = x;
        double y_ = y;
        x = matrix[0][0] * x_ + matrix[0][1] * y_;
        y = matrix[1][0] * x_ + matrix[1][1] * y_;
    }

    double dotProduct(const Direction* other) const {
        return x * other->x + y * other->y;
    }

    void normalize() {
        double magnitude = sqrt(x * x + y * y);
        if (magnitude != 0.0) {
            x /= magnitude;
            y /= magnitude;
        }
    }
};
