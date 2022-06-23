#include "Camera.h"

YRender::Camera::Camera(glm::vec3 position, bool movable)
    :_movable(movable),
    _position(position),
    _world_up(0.0f, 1.0f, 0.0f),
    _front (0.0f, 0.0f, -1.0f)
{
    update_vectors();
}

void YRender::Camera::process_keyboard(CameraMovement dir, float delta_time)
{
    if (!_movable)return;

    float velocity = _move_speed * delta_time;
    if (dir == LEFT)_position -= _right * velocity;
    if (dir == RIGHT)_position += _right * velocity;
    if (dir == FORWARD)_position += _front * velocity;
    if (dir == BACKWARD) _position -= _front * velocity;
}
void YRender::Camera::process_mouse_movement(float xoffset, float yoffset, bool constrain)
{
    if (!_movable)return;

    xoffset *= _mouse_speed;
    yoffset *= _mouse_speed;
    _yaw += xoffset;
    _pitch += yoffset;
    if (constrain)
    {
        if (_pitch > 89.0f)_pitch = 89.0f;
        if (_pitch < -89.0f) _pitch = -89.0f;
    }
    update_vectors();
}




glm::mat4 YRender::Camera::look_at()const {
    return glm::lookAt(_position, _position + _front, _up);
}

void YRender::Camera::update_vectors()
{
    glm::vec3 front;
    front.x = cos(glm::radians(_yaw)) * cos(glm::radians(_pitch));
    front.y = sin(glm::radians(_pitch));
    front.z = sin(glm::radians(_yaw)) * cos(glm::radians(_pitch));
    _front = glm::normalize(front);
    _right = glm::normalize(glm::cross(_front, _world_up));
    _up = glm::normalize(glm::cross(_right, _front));
}