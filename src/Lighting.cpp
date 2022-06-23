#include "Lighting.h"

YRender::Lighting::Lighting(
	glm::vec3 dir, glm::vec3 ambient,
	glm::vec3 diffuse, glm::vec3 specular)
	:_direction(dir),
	_ambient(ambient),
	_diffuse(diffuse), 
	_specular(specular) 
{

}