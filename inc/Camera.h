#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace YRender {

    enum CameraMovement {
        LEFT, RIGHT,
        FORWARD, BACKWARD
    };

    class Camera
    {
    public:
        Camera(glm::vec3 position, bool movable = false);

        glm::mat4 look_at()const;

        void process_keyboard(CameraMovement dir, float delta_time);
        void process_mouse_movement(float xoffset, float yoffset, bool constrain = true);

        ~Camera() = default;
    private:
        void update_vectors();

    public:
        bool _movable = false;

        glm::vec3 _position;
        glm::vec3 _front, _up, _right, _world_up;

        float _yaw = -90.0f, _pitch = 0.0f, _zoom = 45.0f;
        float _move_speed = 1.f, _mouse_speed = 10.f;

    };
}
