cmake_minimum_required(VERSION 3.16)
project(NN-PBR-Vulkan)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Debug symbols for debugging with lldb
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")

# Homebrew paths
set(HOMEBREW_PREFIX /opt/homebrew)

# Find Vulkan
find_package(Vulkan REQUIRED)

# Direct homebrew paths
set(GLFW_INCLUDE_DIR ${HOMEBREW_PREFIX}/Cellar/glfw/3.4/include)
set(GLFW_LIB ${HOMEBREW_PREFIX}/lib/libglfw.dylib)
set(GLM_INCLUDE_DIR ${HOMEBREW_PREFIX}/Cellar/glm/1.0.1/include)

# Shader compilation
find_program(GLSLC glslc REQUIRED)

# Compile shaders
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/shaders)

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/shaders/neural_material_decode.vert.spv
    COMMAND ${GLSLC} ${CMAKE_SOURCE_DIR}/shaders/neural_material_decode.vert
            -o ${CMAKE_BINARY_DIR}/shaders/neural_material_decode.vert.spv
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/neural_material_decode.vert
)

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/shaders/neural_material_decode.frag.spv
    COMMAND ${GLSLC} ${CMAKE_SOURCE_DIR}/shaders/neural_material_decode.frag
            -o ${CMAKE_BINARY_DIR}/shaders/neural_material_decode.frag.spv
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/neural_material_decode.frag
)

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/shaders/neural_material_decode_debug.frag.spv
    COMMAND ${GLSLC} ${CMAKE_SOURCE_DIR}/shaders/neural_material_decode_debug.frag
            -o ${CMAKE_BINARY_DIR}/shaders/neural_material_decode_debug.frag.spv
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/neural_material_decode_debug.frag
)

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/shaders/simple_quad.frag.spv
    COMMAND ${GLSLC} ${CMAKE_SOURCE_DIR}/shaders/simple_quad.frag
            -o ${CMAKE_BINARY_DIR}/shaders/simple_quad.frag.spv
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/simple_quad.frag
)

add_custom_target(compile_shaders ALL
    DEPENDS
        ${CMAKE_BINARY_DIR}/shaders/neural_material_decode.vert.spv
        ${CMAKE_BINARY_DIR}/shaders/neural_material_decode.frag.spv
        ${CMAKE_BINARY_DIR}/shaders/neural_material_decode_debug.frag.spv
        ${CMAKE_BINARY_DIR}/shaders/simple_quad.frag.spv
)

# Main executable
add_executable(vk_neural_demo
    vk_neural_material_demo.cpp
)

add_dependencies(vk_neural_demo compile_shaders)

target_include_directories(vk_neural_demo PRIVATE
    ${Vulkan_INCLUDE_DIRS}
    ${GLFW_INCLUDE_DIR}
    ${GLM_INCLUDE_DIR}
)

# Pass CMAKE_BINARY_DIR to the source code for shader path resolution
target_compile_definitions(vk_neural_demo PRIVATE
    CMAKE_BINARY_DIR="${CMAKE_BINARY_DIR}"
)

target_link_libraries(vk_neural_demo PRIVATE
    Vulkan::Vulkan
    ${GLFW_LIB}
)
