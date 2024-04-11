#pragma once
#include "player.h"
#include "settings.hpp"
#include "coordinate.hpp"

Player::Player(const Map* map) {
    this->coordinate = new Coordinate(PLAYER_START_COORDINATE_X, PLAYER_START_COORDINATE_Y);
    this->direction = new Direction(PLAYER_START_DIRECTION_X, PLAYER_START_DIRECTION_Y);
    this->camera = new Camera(new Direction(-PLAYER_START_DIRECTION_X, PLAYER_START_DIRECTION_Y));
    this->map = map;
}

Player::~Player() {
    delete this->coordinate;
    delete this->direction;
    delete this->camera;
}

void Player::moveX(double dx) {
    // Move according to the camera plane
    Coordinate* newCoordinate = new Coordinate(
        this->coordinate->x + this->camera->direction->x * dx * PLAYER_SPEED,
        this->coordinate->y + this->camera->direction->y * dx * PLAYER_SPEED
    );
    if (!this->map->isWall(newCoordinate)) {
        delete this->coordinate;
        this->coordinate = newCoordinate;
    }
    else delete newCoordinate;
}

void Player::moveY(double dy) {
    // Move according to the player plane
    Coordinate* newCoordinate = new Coordinate(
        this->coordinate->x + this->direction->x * dy * PLAYER_SPEED, 
        this->coordinate->y + this->direction->y * dy * PLAYER_SPEED
    );
    if (!this->map->isWall(newCoordinate)) {
        delete this->coordinate;
        this->coordinate = newCoordinate;
    }
    else delete newCoordinate;
}

void Player::rotate(int mouse_dx) {
    double alfa = -atan(mouse_dx / this->camera->distanceToPlayer) * PLAYER_SENSITIVITY;
    double rotationmatrix[2][2] = {
        {cos(alfa), -sin(alfa)},
        {sin(alfa), cos(alfa)}
    };
    this->direction->dotProduct(rotationmatrix);
    this->camera->direction->dotProduct(rotationmatrix);
}
