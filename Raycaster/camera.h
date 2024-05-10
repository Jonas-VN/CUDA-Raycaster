#pragma once
#include "direction.hpp"


class Camera {
public:
    Direction* direction;
    const int pov;
    const double distanceToPlayer;

public:
    Camera(Direction* direction);
    ~Camera();
    void rotate(const double rotationmatrix[2][2]);
};

