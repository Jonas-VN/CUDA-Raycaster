#pragma once
#include "coordinate.hpp"
#include "direction.hpp"
#include "camera.h"
#include "map.hpp"

class Player {
public:
    Coordinate* coordinate;
    Direction* direction;
    Camera* camera;
    const Map* map;
public:
    Player(const Map* map);
    ~Player();
    void moveX(double dx);
    void moveY(double dy);
    void rotate(int mouse_dx);
};
