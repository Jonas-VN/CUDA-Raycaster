#pragma once
#include "camera.h"
#include "settings.hpp"
#include <cmath>

Camera::Camera(Direction* direction) {
    this->direction = direction;
    this->pov = POV;
    this->distanceToPlayer = atan(POV * DEG2RAD);
}

Camera::~Camera() {
    delete direction;
}

void Camera::rotate(const double rotationmatrix[2][2]) {
    direction->dotProduct(rotationmatrix);
}
