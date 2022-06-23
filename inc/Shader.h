#pragma once
#include "glad.h"
#include <glm/glm.hpp>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

namespace YRender {
	class Shader
	{
	public:
		Shader(const std::string& vertex_path, const std::string& fragment_path, const std::string& geometry_path = "");

		void use()const { glUseProgram(_id); }
		void set_bool(const std::string& name, bool value) const;
		void set_int(const std::string& name, int value) const;
		void set_float(const std::string& name, float value) const;
		void set_vec3(const std::string& name, const glm::vec3& value) const;
		void set_mat4(const std::string& name, const glm::mat4& mat) const;

		std::string _v_path, _f_path, _g_path;
	private:
		void check_compile_errors(GLuint shader, std::string type);

	private:
		unsigned int _id;
	};
}
