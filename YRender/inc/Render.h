#pragma once
#include <glm/glm.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

#include <vector>
#include "Shader.h"
#include "Camera.h"
#include "Lighting.h"

namespace YRender {
	enum {
		Y_POINTS = 0x0000, Y_LINES = 0x0001, Y_LINE_LOOP = 0x0002, Y_LINE_STRIP = 0x0003,
		Y_TRIANGLES = 0x0004, Y_TRIANGLE_STRIP = 0x0005, Y_TRIANGLE_FAN = 0x0006, Y_QUADS = 0x0007,

		Y_FRONT = 0x0404, Y_BACK = 0x0405, Y_LEFT = 0x0406,Y_RIGHT = 0x0407, Y_FRONT_AND_BACK = 0x0408,

		Y_LINE = 0x1B01, Y_FILL = 0x1B02, 
#ifndef USER_DEFINED_SHADER
		Y_LINE_AND_FILL = 0x1B03,
#endif
		Y_POLYGON_OFFSET_FILL = 0x8037, Y_PROGRAM_POINT_SIZE = 0x8642
	};
	struct Vertex {
		glm::vec3 _position;
		glm::vec3 _normal;
		glm::vec3 _attr1 = glm::vec3(0.);
		glm::vec3 _attr2 = glm::vec3(0.);
		glm::vec3 _attr3 = glm::vec3(0.);
		float _attr4 = 0.; 
		float _attr5 = 0.;
		float _attr6 = 0.;
	};

	struct ObjectInfo
	{
		long long _v_count;
		long long _i_count;
		int _type;
		glm::mat4 _trans = glm::mat4(1);
		int _polygon_mode = Y_FILL;
		glm::vec3 _color = glm::vec3(1.);
	};

	class Render
	{
	public:
		Render();
		~Render();

#ifdef USER_DEFINED_SHADER
		void render(unsigned fbo, bool clear, const Shader& shader);
#else
		void render(unsigned fbo, bool clear, int width, int height, const glm::mat4& model_trans);
#endif //USER_DEFINED_SHADER

		int push_object(
			const Eigen::Matrix3Xd& V, const Eigen::MatrixXi& CELLS, 
			int draw_typ,const glm::mat4& trans,
			std::initializer_list<Eigen::Matrix3Xd> mat_attr,
			std::initializer_list<Eigen::VectorXd> float_attr);

		void update_object(
			int obj_id,
			const Eigen::Matrix3Xd& V, const Eigen::MatrixXi& CELLS,
			int draw_typ, const glm::mat4& trans,
			std::initializer_list<Eigen::Matrix3Xd> mat_attr,
			std::initializer_list<Eigen::VectorXd> float_attr);

		void delete_object(int obj_id);
		void clear_object();

#ifndef USER_DEFINED_SHADER
		void set_object_color(int obj_id ,glm::vec3 color);
#endif
		void set_object_polygon_mode(int obj_id, int mode);

	private:
		void calc_vertex_normal(const Eigen::Matrix3Xd& V, const Eigen::Matrix3Xi& F, Eigen::Matrix3Xd& normals);
	
	private:
		unsigned int _vao, _vbo, _ebo;
		long long _i_count=0, _v_count=0;
		std::vector<ObjectInfo> _objects;

		Shader _shader = Shader("shader/default.vs", "shader/default.fs");
		Camera 	_camera = Camera(glm::vec3(0.0f, 0.0f, 2.0f), false);
		Lighting _lighting = Lighting(glm::vec3(-2.2f, -1.0f, -2.3f));
	};
}


