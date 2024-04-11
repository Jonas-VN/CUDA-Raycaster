#pragma once
#include "direction.hpp"


struct Coordinate {
    double x;
    double y;

    Coordinate(double x, double y) : x(x), y(y) {}
    Coordinate() : x(0), y(0) {}
};