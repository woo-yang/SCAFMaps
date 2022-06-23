cmake_minimum_required(VERSION 2.8)

project(SCAFMaps)
set(CMAKE_CXX_STANDARD 17)

FILE(GLOB SOURCES src/*.cpp)

set(YRender "D:/repos/YRender")
INCLUDE_DIRECTORIES(
    ${YRender}/inc
    $ENV{INCLUDE_DIR}
    $ENV{INCLUDE_DIR}/Eigen
)


link_directories(
    $ENV{LIB_DIR}/triangle
)

add_subdirectory(common)
add_subdirectory(io)
add_subdirectory(alg)
add_subdirectory(solver)
add_subdirectory(${YRender} YRender_binary_dir)

add_executable(SCAFMaps ${SOURCES})

target_link_libraries(SCAFMaps PUBLIC

    YRender
    common
    mesh_io
    algorithm
    )
