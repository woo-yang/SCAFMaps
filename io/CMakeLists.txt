file(GLOB header *.h *.hpp)
file(GLOB source *.cpp)


if(MSVC)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
    set(BUILD_SHARED_LIBS TRUE)
endif()

add_library(mesh_io SHARED ${source} ${header})
SET_TARGET_PROPERTIES(mesh_io PROPERTIES RUNTIME_OUTPUT_DIRECTORY ../)

target_link_libraries(mesh_io
common
)