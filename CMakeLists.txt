cmake_minimum_required(VERSION 3.16)
project(NN-PBR-Vulkan)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find packages
find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)

# Shader compilation
find_program(GLSLC glslc)
if(NOT GLSLC)
    message(FATAL_ERROR "glslc not found. Install the Vulkan SDK.")
endif()

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

add_custom_target(compile_shaders ALL
    DEPENDS
        ${CMAKE_BINARY_DIR}/shaders/neural_material_decode.vert.spv
        ${CMAKE_BINARY_DIR}/shaders/neural_material_decode.frag.spv
)

# Main executable
add_executable(vk_neural_demo
    vk_neural_material_demo.cpp
)

add_dependencies(vk_neural_demo compile_shaders)

target_include_directories(vk_neural_demo PRIVATE
    ${Vulkan_INCLUDE_DIRS}
    ${GLM_INCLUDE_DIRS}
)

target_link_libraries(vk_neural_demo PRIVATE
    Vulkan::Vulkan
    glfw
    glm::glm
)

# Copy shaders to build directory
add_custom_command(TARGET vk_neural_demo POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_BINARY_DIR}/shaders
        ${CMAKE_BINARY_DIR}/shaders
)
