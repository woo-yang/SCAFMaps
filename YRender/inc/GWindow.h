#pragma once
#include "glad.h"
#include <GLFW/glfw3.h>

#include <memory>
#include <string>
#include <iostream>

#include "Render.h"

namespace YRender {

	class GWindow
	{
	public:
		GWindow(std::string title, double width = 1200, double height = 600);
		~GWindow();

		void show();
		void refresh(unsigned fbo = 0, bool clear = true);

		int push_object(
			const Eigen::Matrix3Xd& V, const Eigen::MatrixXi& CELLS,
			int draw_typ, bool normalized = false,
			std::initializer_list<Eigen::Matrix3Xd> mat_attr = {},
			std::initializer_list<Eigen::VectorXd> float_attr = {});
		void update_object(
			int obj_id,
			const Eigen::Matrix3Xd& V, const Eigen::MatrixXi& CELLS, 
			int draw_typ, bool normalized = false,
			std::initializer_list<Eigen::Matrix3Xd> mat_attr = {},
			std::initializer_list<Eigen::VectorXd> float_attr = {});

		void delete_object(int obj_id) { _render->delete_object(obj_id); }

		void clear_object() { _render->clear_object(); }

#ifdef USER_DEFINED_SHADER
		void set_shader(
			const std::string& vertex_path,
			const std::string& fragment_path,
			const std::string& geometry_path = "");

		void set_shader_attri(const std::string& name, int value);
		void set_shader_attri(const std::string& name, float value);
		void set_shader_attri(const std::string& name, const Eigen::Vector3d& value);
		void set_shader_attri(const std::string& name, const Eigen::Matrix4d& value);
#else
		void set_object_color(int id, double r, double g, double b);
#endif // USER_DEFINED_SHADER
		
		void set_object_polygon_mode(int id , int mode);

	private:
		void  normalized_vertex(const Eigen::Matrix3Xd& V,Eigen::Vector3d& pos, double& scale);

	protected:
		Eigen::Matrix4d get_trans_mat4()const;
		virtual void mouse_button_callback(GLFWwindow* window, int button, int action, int modifiers);
		virtual void cursor_pos_callback(GLFWwindow* window, double xpos, double ypos);
		virtual void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
		virtual void keyboard_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
		virtual void framebuffer_size_callback(GLFWwindow* window, int width, int height);

	protected:
		GLFWwindow* _window;

		std::shared_ptr<YRender::Render> _render;
		std::shared_ptr<YRender::Shader> _shader;

		double _width, _height;

		glm::vec3 _position = glm::vec3(0, 0, 0);
		glm::vec3 _scale = glm::vec3(1.0, 1.0, 1.0);
		glm::mat4 _rotate = glm::mat4(1.0f);

		double _left_x, _left_y, _right_x, _right_y;
		bool _left_first = true, _right_first = true;
		double _last_frame = 0., _delta_time = 0., _mouse_speed = 0.005;
	};

}

