file(GLOB header *.h *.hpp)
file(GLOB source *.cpp)


if(MSVC)
    add_definitions(-D_USE_MATH_DEFINES)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
    set(BUILD_SHARED_LIBS TRUE)
endif()

add_library(solver SHARED ${source} ${header})
SET_TARGET_PROPERTIES(solver PROPERTIES RUNTIME_OUTPUT_DIRECTORY ../)

target_link_libraries(solver

)