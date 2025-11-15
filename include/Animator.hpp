#pragma once

#include <tiny_gltf.h>

#include "Scene.hpp"

class Animator {
public:
    void animate(const tinygltf::Model& model, Scene& scene, float time);
};
