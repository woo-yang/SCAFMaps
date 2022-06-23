#pragma once
#include <glm/glm.hpp>

namespace YRender {
    class Lighting
    {
    public:
        Lighting(glm::vec3 dir,
            glm::vec3 ambient = glm::vec3(0.3f, 0.3f, 0.3f),
            glm::vec3 diffuse = glm::vec3(0.8f, 0.8f, 0.8f),
            glm::vec3 specular = glm::vec3(0.5f, 0.5f, 0.5f));

        ~Lighting() = default;

    public:
        glm::vec3 _direction;
        glm::vec3 _ambient;
        glm::vec3 _diffuse;
        glm::vec3 _specular;
    };

}

