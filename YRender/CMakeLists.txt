cmake_minimum_required(VERSION 3.0)

project(YRender)
set(CMAKE_CXX_STANDARD 11)

option(USER_DEFINED_SHADER "user defined shader" OFF)
if(USER_DEFINED_SHADER)
add_definitions(-DUSER_DEFINED_SHADER)
endif()

file(GLOB header inc/*.h)
file(GLOB source src/*.cpp src/*.c)

find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)

configure_file(shader/default.vs ${CMAKE_BINARY_DIR}/shader/default.vs COPYONLY)
configure_file(shader/default.fs ${CMAKE_BINARY_DIR}/shader/default.fs COPYONLY)


add_library(${PROJECT_NAME} ${header} ${source})
#add_executable(${PROJECT_NAME} ${header} ${source})


target_include_directories(${PROJECT_NAME}
	PUBLIC 
	inc
    ${OPENGL_INCLUDE_DIR}
	$ENV{INCLUDE_DIR}
	$ENV{INCLUDE_DIR}/Eigen
    )
if(UNIX)
	target_link_libraries(${PROJECT_NAME}
	PUBLIC
	dl
	)
endif()

target_link_libraries(${PROJECT_NAME}
    PUBLIC
    ${OPENGL_gl_LIBRARY}
	glfw
)