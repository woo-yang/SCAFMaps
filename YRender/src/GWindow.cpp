#include "GWindow.h"
#include "Render.h"
#include "EGLM.h"
#include <iostream>


YRender::GWindow::GWindow(std::string title, double width , double height )
    :_width(width), _height(height), _left_x(_width / 2.f),_left_y(_height/2.f)
{
    /*---------------- init window ------------------*/
    glfwInit();
    glfwSetTime(0);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    _window = glfwCreateWindow(_width, _height, title.c_str(), nullptr, nullptr);

    if (_window == nullptr) {
        std::cerr << "Failed to create GLFW window!" << std::endl;
        glfwTerminate();
        return;
    }
    glfwMakeContextCurrent(_window);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
        throw std::runtime_error("Could not initialize GLAD!");

    glfwSetWindowUserPointer(_window, this);

    auto mouse_button_func = [](GLFWwindow* w, int button, int action, int mod)
    {
        static_cast<GWindow*>(glfwGetWindowUserPointer(w))->mouse_button_callback(w, button, action, mod);
    };

    glfwSetMouseButtonCallback(_window, mouse_button_func);

    auto cursor_pos_func = [](GLFWwindow* w, double xpos, double ypos)
    {
        static_cast<GWindow*>(glfwGetWindowUserPointer(w))->cursor_pos_callback(w, xpos, ypos);
    };

    glfwSetCursorPosCallback(_window, cursor_pos_func);

    auto scroll_func = [](GLFWwindow* w, double xoffset, double yoffset)
    {
        static_cast<GWindow*>(glfwGetWindowUserPointer(w))->scroll_callback(w, xoffset, yoffset);
    };

    glfwSetScrollCallback(_window, scroll_func);

    auto framebuffer_func = [](GLFWwindow* w, int width, int height)
    {
        static_cast<GWindow*>(glfwGetWindowUserPointer(w))->framebuffer_size_callback(w, width, height);
    };

    glfwSetFramebufferSizeCallback(_window, framebuffer_func);

    auto keyboard_func = [](GLFWwindow* w, int key, int scancode, int action, int mods)
    {
        static_cast<GWindow*>(glfwGetWindowUserPointer(w))->keyboard_callback(w, key, scancode, action, mods);
    };

    glfwSetKeyCallback(_window, keyboard_func);
    
    /*---------------- init window finish ------------------*/

    _render = std::make_shared<YRender::Render>();

}
void YRender::GWindow::show() {
    glfwMakeContextCurrent(_window);
    while (!glfwWindowShouldClose(_window)) {

        _delta_time = glfwGetTime() - _last_frame;
        _last_frame = glfwGetTime();

        glfwPollEvents();
#ifdef USER_DEFINED_SHADER
        assert(_shader != nullptr);
        _render->render(0, true, *_shader);
#else
        _render->render(0, true, _width, _height, E2GLM<double, 4, 4>(get_trans_mat4()));
#endif // USER_DEFINED_SHADER

        glfwSwapBuffers(_window);
    }
    glfwTerminate();
}

void YRender::GWindow::refresh(unsigned fbo, bool clear) {
    glfwMakeContextCurrent(_window);
    _delta_time = glfwGetTime() - _last_frame;
    _last_frame = glfwGetTime();

    glfwPollEvents();
#ifdef USER_DEFINED_SHADER
    assert(_shader != nullptr);
    _render->render(fbo, clear, *_shader);
#else
    _render->render(fbo, clear, _width, _height, E2GLM<double, 4, 4>(get_trans_mat4()));
#endif // USER_DEFINED_SHADER


    glfwSwapBuffers(_window);
}

int YRender::GWindow::push_object(
    const Eigen::Matrix3Xd& V, const Eigen::MatrixXi& CELLS,
    int draw_typ, bool normalized,
    std::initializer_list<Eigen::Matrix3Xd> mat_attr,
    std::initializer_list<Eigen::VectorXd> float_attr) {

    Eigen::Vector3d pos = Eigen::Vector3d::Zero();
    double scale = 1;
    if (normalized) normalized_vertex(V, pos, scale);
    glm::mat4 m(1.0f);
    m = glm::scale(m, glm::vec3(scale));
    m = glm::translate(m, E2GLM(pos));
    return _render->push_object(V, CELLS, draw_typ, m, mat_attr, float_attr);
}

void YRender::GWindow::update_object(
    int obj_id,
    const Eigen::Matrix3Xd& V, const Eigen::MatrixXi& CELLS, 
    int draw_typ, bool normalized,
    std::initializer_list<Eigen::Matrix3Xd> mat_attr,
    std::initializer_list<Eigen::VectorXd> float_attr) 
{
    Eigen::Vector3d pos = Eigen::Vector3d::Zero();
    double scale = 1;
    if (normalized) normalized_vertex(V, pos, scale);
    glm::mat4 m(1.0f);
    m = glm::scale(m, glm::vec3(scale));
    m = glm::translate(m, E2GLM(pos));
    _render->update_object(obj_id, V, CELLS, draw_typ, m, mat_attr, float_attr);
}

void  YRender::GWindow::normalized_vertex(
    const Eigen::Matrix3Xd& V ,
    Eigen::Vector3d& pos,double& scale) 
{

    Eigen::Vector3d center = V.rowwise().sum();
    Eigen::Vector3d max = V.rowwise().maxCoeff();
    Eigen::Vector3d min = V.rowwise().minCoeff();

    center /= V.cols();
    double max_len = (max - min).maxCoeff();

    pos = -center;
    scale = 1. / max_len;

}

Eigen::Matrix4d YRender::GWindow::get_trans_mat4()const {
    glm::mat4 m(1.0f);
    m *= _rotate;
    m = glm::scale(m, _scale);
    m = glm::translate(m, _position);
    return GLM2E<double, 4, 4>(m);
}

void YRender::GWindow::mouse_button_callback(GLFWwindow* window, int button, int action, int modifiers) {

    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE)
        _left_first = true;
    if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_RELEASE)
        _right_first = true;
}

void YRender::GWindow::cursor_pos_callback(GLFWwindow* window, double xpos, double ypos) {

    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)) {
        //Called when the left button is pressed and dragged
        if (_left_first) {
            _left_x = xpos; _left_y = ypos;
            _left_first = false;
        }
        float xoffset = xpos - _left_x, yoffset = _left_y - ypos;
        _left_x = xpos; _left_y = ypos;
        xoffset *= _mouse_speed;
        yoffset *= _mouse_speed;

        glm::mat4 m(1.0f);
        m = glm::rotate(m, xoffset, { 0.0f, 1.0f, 0.0f });
        _rotate = m * _rotate;
        m = glm::mat4(1.0f);
        m = glm::rotate(m, yoffset, { -1.0f, 0.0f, 0.0f });
        _rotate = m * _rotate;

    }
    else if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT)) {
        if (_right_first) {
            _right_x = xpos; _right_y = ypos;
            _right_first = false;
        }
        float xoffset = xpos - _right_x, yoffset = _right_y - ypos;
        _right_x = xpos; _right_y = ypos;
        //no operate
    }

}
void YRender::GWindow::scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    _scale += glm::vec3(yoffset * 0.1* _scale.x);
    if (_scale.x < 1e-3 || _scale.y < 1e-3 || _scale.z < 1e-3)
        _scale = glm::vec3(1e-3);
}

void YRender::GWindow::keyboard_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

}
void YRender::GWindow::framebuffer_size_callback(GLFWwindow* window, int width, int height) {

    _width = width;
    _height = height;
    glViewport(0, 0, width, height);
}

#ifdef USER_DEFINED_SHADER
void YRender::GWindow::set_shader(const std::string& vertex_path, const std::string& fragment_path, const std::string& geometry_path ) {
    _shader = std::make_shared<Shader>(vertex_path, fragment_path, geometry_path);
    _shader->use();
}

void YRender::GWindow::set_shader_attri(const std::string& name, int value) {
    _shader->set_int(name, value);
}
void YRender::GWindow::set_shader_attri(const std::string& name, float value) {
    _shader->set_float(name, value);
}
void YRender::GWindow::set_shader_attri(const std::string& name, const Eigen::Vector3d& value) {
    _shader->set_vec3(name, E2GLM<double,3>(value));
}
void YRender::GWindow::set_shader_attri(const std::string& name, const Eigen::Matrix4d& value) {
    _shader->set_mat4(name, E2GLM<double, 4, 4>(value));
}
#else
void YRender::GWindow::set_object_color(int id, double r, double g, double b)
{

    _render->set_object_color(id, { r,g,b });
}

#endif // USER_DEFINED_SHADER


void YRender::GWindow::set_object_polygon_mode(int id, int mode) {

    _render->set_object_polygon_mode(id, mode);
}


YRender::GWindow::~GWindow() {
    glfwTerminate();
}