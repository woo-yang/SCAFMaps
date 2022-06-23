#include "Shader.h"
#include <type_traits>

YRender::Shader::Shader(
	const std::string& vertex_path,
	const std::string& fragment_path,
	const std::string& geometry_path)
{

	this->_v_path = vertex_path;
	this->_f_path = fragment_path;
	this->_g_path = geometry_path;

	std::ifstream v_file, f_file, g_file;
	std::ostringstream v_stream, f_stream,g_stream;

	try {
		v_file.open(vertex_path);
		f_file.open(fragment_path);
		
		v_stream << v_file.rdbuf();
		f_stream << f_file.rdbuf();

		v_file.close();
		f_file.close();

		if (geometry_path != "")
		{
			g_file.open(geometry_path);
			g_stream << g_file.rdbuf();
			g_file.close();
		}
	}
	catch (std::ifstream::failure e)
	{
		std::cout << "Shader file not succesfully read!" << std::endl;
		std::cout << e.what() << std::endl;
	}
	std::string temp = v_stream.str();
	const char* v_code = temp.c_str();
	std::string temp2 = f_stream.str();
	const char* f_code = temp2.c_str();

	unsigned int vertex_shader, fragment_shader, geometry_shader;

	vertex_shader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertex_shader, 1, &v_code, NULL);
	glCompileShader(vertex_shader);
	check_compile_errors(vertex_shader, "VERTEX");

	fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragment_shader, 1, &f_code, NULL);
	glCompileShader(fragment_shader);
	check_compile_errors(fragment_shader, "FRAGMENT");

	if (geometry_path != "") {
		std::string temp = g_stream.str();
		const char* g_code = temp.c_str();
		geometry_shader = glCreateShader(GL_GEOMETRY_SHADER);
		glShaderSource(geometry_shader, 1, &g_code, NULL);
		glCompileShader(geometry_shader);
		check_compile_errors(geometry_shader, "GEOMETRY");
	}

	_id = glCreateProgram();
	glAttachShader(_id, vertex_shader);
	glAttachShader(_id, fragment_shader);
	if (geometry_path != "")glAttachShader(_id, geometry_shader);
	glLinkProgram(_id);
	check_compile_errors(_id, "PROGRAM");

	glDeleteShader(vertex_shader);
	glDeleteShader(fragment_shader);
	if (geometry_path != "")glDeleteShader(geometry_shader);
}


void YRender::Shader::set_bool(const std::string& name, bool value)const {
	glUniform1i(glGetUniformLocation(_id, name.c_str()), value);
}

void YRender::Shader::set_int(const std::string& name, int value) const {
	glUniform1i(glGetUniformLocation(_id, name.c_str()), value);
}
void YRender::Shader::set_float(const std::string& name, float value) const {
	glUniform1f(glGetUniformLocation(_id, name.c_str()), value);
}

void YRender::Shader::set_vec3(const std::string& name, const glm::vec3& value) const {
	glUniform3fv(glGetUniformLocation(_id, name.c_str()), 1, &value[0]);
}

void YRender::Shader::set_mat4(const std::string& name, const glm::mat4& mat) const {

	glUniformMatrix4fv(glGetUniformLocation(_id, name.c_str()), 1, GL_FALSE, &mat[0][0]);
}

void YRender::Shader::check_compile_errors(GLuint shader, std::string type) {
	GLint success;
	GLchar infoLog[1024];
	if (type != "PROGRAM") {
		glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
		if (!success)
		{
			glGetShaderInfoLog(shader, 1024, NULL, infoLog);
			std::cout << "Shader compilation error of type: " << type << "\n"<< infoLog <<  std::endl;
		}
	}
	else {
		glGetProgramiv(shader, GL_LINK_STATUS, &success);
		if (!success)
		{
			glGetProgramInfoLog(shader, 1024, NULL, infoLog);
			std::cout << "Program linking error of type: " << type << "\n"<< infoLog << std::endl;
		}
	}
}

