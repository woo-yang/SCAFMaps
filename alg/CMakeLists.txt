file(GLOB header *.h)
file(GLOB source *.cpp)

if(MSVC)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
    set(BUILD_SHARED_LIBS TRUE)
endif()

INCLUDE_DIRECTORIES(
    $ENV{INCLUDE_DIR}
    $ENV{INCLUDE_DIR}/igl
)

add_library(algorithm SHARED ${header} ${source})
SET_TARGET_PROPERTIES(algorithm PROPERTIES RUNTIME_OUTPUT_DIRECTORY ../)

target_link_libraries(algorithm
    common
    solver
    debug triangled optimized triangle
)
