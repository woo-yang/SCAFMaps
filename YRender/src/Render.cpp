#include "Render.h"
#include "EGLM.h"
#include <iostream>
#include <initializer_list>

YRender::Render::Render()
{
    glGenVertexArrays(1, &_vao);
    glGenBuffers(1, &_vbo);
    glGenBuffers(1, &_ebo);

    glBindVertexArray(_vao);
    glBindBuffer(GL_ARRAY_BUFFER, _vbo);
    glBufferData(GL_ARRAY_BUFFER, 1e8 * sizeof(Vertex), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 1e8 * sizeof(unsigned int), nullptr, GL_DYNAMIC_DRAW);

    std::vector<int> v_struct{ 3,3,3,3,3,1,1,1 };
    int offset = 0;
    for (int i = 0; i < v_struct.size(); ++i) {
        glEnableVertexAttribArray(i);
        glVertexAttribPointer(i, v_struct[i], GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offset);
        offset += v_struct[i] * sizeof(float);
    }
    glBindVertexArray(0);
}

#ifdef USER_DEFINED_SHADER
void YRender::Render::render(unsigned fbo, bool clear, const Shader& shader) {
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glEnable(GL_DEPTH_TEST);
    if (clear) {
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }
    shader.use();
    glBindVertexArray(_vao);
    int offset = 0;
    for (int i = 0; i < _objects.size(); ++i) {
        if (_objects[i]._type != -1) {
            glPolygonMode(Y_FRONT_AND_BACK, _objects[i]._polygon_mode);
            glDrawElements(_objects[i]._type, _objects[i]._i_count, GL_UNSIGNED_INT, (void*)offset);

        }
        offset += _objects[i]._i_count * sizeof(unsigned);
    }
    glBindVertexArray(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}
#else
void YRender::Render::render(unsigned fbo, bool clear, int width, int height, const glm::mat4& model_trans) {

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glEnable(GL_DEPTH_TEST);
    if (clear) {
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }
    _shader.use();
    glm::mat4 view = _camera.look_at();
    glm::mat4 projection = glm::perspective<float>(glm::radians(_camera._zoom), (float)width / height, 0.1f, 100.0f);

    _shader.set_mat4("view", view);
    _shader.set_mat4("projection", projection);
    _shader.set_vec3("direction", _camera._front);
    _shader.set_mat4("whole_model", model_trans);

    _shader.set_vec3("viewPos", _camera._position);
    _shader.set_vec3("dirLight.direction", _lighting._direction);
    _shader.set_vec3("dirLight.ambient", _lighting._ambient);
    _shader.set_vec3("dirLight.diffuse", _lighting._diffuse);
    _shader.set_vec3("dirLight.specular", _lighting._specular);

    glBindVertexArray(_vao);
    int offset = 0;
    for (int i = 0; i < _objects.size(); ++i) {
        if (_objects[i]._type != -1) {
            _shader.set_mat4("model", _objects[i]._trans);
            _shader.set_vec3("objColor", _objects[i]._color);
            if (_objects[i]._polygon_mode == Y_LINE || _objects[i]._polygon_mode == Y_FILL) {
                glPolygonMode(Y_FRONT_AND_BACK, _objects[i]._polygon_mode);
                glDrawElements(_objects[i]._type, _objects[i]._i_count, GL_UNSIGNED_INT, (void*)offset);
            }
            else if(_objects[i]._polygon_mode == Y_LINE_AND_FILL){
                glPolygonMode(Y_FRONT_AND_BACK, Y_FILL);
                glDrawElements(_objects[i]._type, _objects[i]._i_count, GL_UNSIGNED_INT, (void*)offset);
                _shader.set_vec3("objColor", { 0,0,0 });
                glEnable(GL_POLYGON_OFFSET_FILL);
                glPolygonOffset(1, 1);
                glPolygonMode(Y_FRONT_AND_BACK, Y_LINE);
                glDrawElements(_objects[i]._type, _objects[i]._i_count, GL_UNSIGNED_INT, (void*)offset);
            }
        }
        offset += _objects[i]._i_count * sizeof(unsigned);
    }
    glBindVertexArray(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

#endif //USER_DEFINED_SHADER

void YRender::Render::calc_vertex_normal(const Eigen::Matrix3Xd& V, const Eigen::Matrix3Xi& F, Eigen::Matrix3Xd& normals)
{
    normals.setZero();
    for (int i = 0; i < F.cols(); ++i) {
        Eigen::Vector3d v1 = V.col(F(0, i)) - V.col(F(1, i));
        Eigen::Vector3d v2 = V.col(F(0, i)) - V.col(F(2, i));
        auto n = v1.cross(v2);
        for (int j = 0; j < 3; ++j) {
            normals.col(F(j, i)) += n;
        }
    }
    normals.colwise().normalize();
}

int YRender::Render::push_object(
    const Eigen::Matrix3Xd& V, const Eigen::MatrixXi& CELLS, 
    int draw_type,const glm::mat4& trans,
    std::initializer_list<Eigen::Matrix3Xd> mat_attr ,
    std::initializer_list<Eigen::VectorXd> float_attr) 
{

    Eigen::MatrixXi cells = CELLS.array() + _v_count;
    Eigen::Matrix3Xd normals(3, V.cols());
    if (CELLS.rows() == 3) calc_vertex_normal(V, CELLS, normals);
    else normals.setZero();
    Vertex v;
    std::vector<Vertex> vertices;

    for (int i = 0; i < V.cols(); ++i) {
        v._position = E2GLM<double, 3>(V.col(i));
        v._normal = E2GLM<double, 3>(normals.col(i));

        if (mat_attr.size() > 0)v._attr1 = E2GLM<double, 3>((*mat_attr.begin()).col(i));
        if (mat_attr.size() > 1)v._attr2 = E2GLM<double, 3>((*(mat_attr.begin() + 1)).col(i));
        if (mat_attr.size() > 2)v._attr3 = E2GLM<double, 3>((*(mat_attr.begin() + 2)).col(i));
        if (float_attr.size() > 0)v._attr4 = (*float_attr.begin())(i);
        if (float_attr.size() > 1)v._attr5 = (*(float_attr.begin() + 1))(i);
        if (float_attr.size() > 2)v._attr6 = (*(float_attr.begin() + 2))(i);

        vertices.emplace_back(v);
    }

    glBindVertexArray(_vao);
    glBindBuffer(GL_ARRAY_BUFFER, _vbo);
    glBufferSubData(GL_ARRAY_BUFFER, _v_count * sizeof(Vertex), vertices.size() * sizeof(Vertex), &vertices[0]);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _ebo);
    glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, _i_count * sizeof(int), CELLS.rows() * CELLS.cols() * sizeof(int), cells.data());

    glBindVertexArray(0);

    _i_count += CELLS.rows() * CELLS.cols();
    _v_count += V.cols();
    _objects.push_back({ V.cols(), CELLS.rows() * CELLS.cols() , draw_type ,trans });

    return _objects.size();
}

void YRender::Render::update_object(
    int obj_id,
    const Eigen::Matrix3Xd& V, const Eigen::MatrixXi& CELLS,
    int draw_type, const glm::mat4& trans,
    std::initializer_list<Eigen::Matrix3Xd> mat_attr,
    std::initializer_list<Eigen::VectorXd> float_attr)
{

    int id = obj_id - 1;
    if (id >= _objects.size() || id < 0)return;

    if (_objects[id]._v_count != V.cols()) {
        std::cerr << "Error number of object vertices!" << std::endl;
        return;
    }
    if (_objects[id]._i_count != 3 * CELLS.cols()) {
        std::cerr << "Error number of object cells!" << std::endl;
        return;
    }

    Eigen::Matrix3Xd normals(3, V.cols());
    if (CELLS.rows() == 3) calc_vertex_normal(V, CELLS, normals);
    else normals.setZero();

    Vertex v;
    std::vector<Vertex> vertices;
    for (int i = 0; i < V.cols(); ++i) {
        v._position = E2GLM<double, 3>(V.col(i));
        v._normal = E2GLM<double, 3>(normals.col(i));
        if (mat_attr.size() > 0)v._attr1 = E2GLM<double, 3>((*mat_attr.begin()).col(i));
        if (mat_attr.size() > 1)v._attr2 = E2GLM<double, 3>((*(mat_attr.begin() + 1)).col(i));
        if (mat_attr.size() > 2)v._attr3 = E2GLM<double, 3>((*(mat_attr.begin() + 2)).col(i));
        if (float_attr.size() > 0)v._attr4 = (*float_attr.begin())(i);
        if (float_attr.size() > 1)v._attr5 = (*(float_attr.begin() + 1))(i);
        if (float_attr.size() > 2)v._attr6 = (*(float_attr.begin() + 2))(i);
        vertices.emplace_back(v);
    }
    int offset = 0;
    for (int i = 0; i < id; ++i) {
        offset += _objects[i]._v_count * sizeof(Vertex);
    }
    glBindVertexArray(_vao);
    glBindBuffer(GL_ARRAY_BUFFER, _vbo);
    glBufferSubData(GL_ARRAY_BUFFER, offset, vertices.size() * sizeof(Vertex), &vertices[0]);
    glBindVertexArray(0);

    _objects[id]._type = draw_type;
    _objects[id]._trans = trans;
}

void YRender::Render::delete_object(int obj_id) {

    int id = obj_id - 1;
    if (id >= _objects.size() || id < 0)return;
    _objects[id]._type = -1;

}

void YRender::Render::clear_object() {

    for (int i = 0; i < _objects.size(); ++i) {
        _objects[i]._type = -1;
    }
}

#ifndef USER_DEFINED_SHADER
void YRender::Render::set_object_color(int obj_id, glm::vec3 color) 
{
    int id = obj_id - 1;
    if (id >= _objects.size() || id < 0)return;
    _objects[id]._color = color;

}
#endif

void YRender::Render::set_object_polygon_mode(int obj_id, int mode) {
    int id = obj_id - 1;
    if (id >= _objects.size() || id < 0)return;
    _objects[id]._polygon_mode = mode;
}

YRender::Render::~Render() {

    glDeleteVertexArrays(1, &_vao);
    glDeleteBuffers(1, &_vbo);
    glDeleteBuffers(1, &_ebo);
}