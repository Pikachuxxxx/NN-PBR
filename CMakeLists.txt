cmake_minimum_required(VERSION 3.16)
project(NN-PBR-Vulkan)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)



# Find Vulkan (Relies on VULKAN_SDK environment variable or system paths)
find_package(Vulkan REQUIRED)
find_program(GLSLC glslc REQUIRED)

# ==============================================================================
# Dependency Fetching (Cross-Platform)
# ==============================================================================
include(FetchContent)

# Fetch GLFW automatically
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.4
)
# Disable unnecessary GLFW build options to speed up compile time
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

# Fetch GLM automatically (Header-only math library)
FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG        1.0.1
)
set(GLM_BUILD_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glm)

# ==============================================================================
# Shader Compilation
# ==============================================================================
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/shaders)

set(SHADER_FILES
    neural_material_decode.vert
    neural_material_decode.frag
    neural_material_decode_debug.frag
)

set(SPV_FILES "")

foreach(SHADER ${SHADER_FILES})
    add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/shaders/${SHADER}.spv
        COMMAND ${GLSLC} ${CMAKE_CURRENT_SOURCE_DIR}/demo/shaders/${SHADER} -o ${CMAKE_BINARY_DIR}/shaders/${SHADER}.spv
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/demo/shaders/${SHADER} ${CMAKE_CURRENT_SOURCE_DIR}/demo/shaders/material_decoder.glsl
    )
    list(APPEND SPV_FILES ${CMAKE_BINARY_DIR}/shaders/${SHADER}.spv)
endforeach()

add_custom_target(compile_shaders ALL DEPENDS ${SPV_FILES})

# ==============================================================================
# Main Executable
# ==============================================================================
add_executable(vk_neural_demo
    demo/vk_neural_material_demo.cpp
)

add_dependencies(vk_neural_demo compile_shaders)

# Include directories (Using the fetched target paths)
target_include_directories(vk_neural_demo PRIVATE
    ${Vulkan_INCLUDE_DIRS}
    ${glm_SOURCE_DIR} 
)

# Pass CMAKE_BINARY_DIR to the source code for shader path resolution
target_compile_definitions(vk_neural_demo PRIVATE
    NNPBR_SAMPLE_EXPORT="${CMAKE_CURRENT_SOURCE_DIR}/models/red-plaid-64/export"
    CMAKE_BINARY_DIR="${CMAKE_BINARY_DIR}"
)

# Link libraries (Vulkan and our fetched GLFW target)
target_link_libraries(vk_neural_demo PRIVATE
    Vulkan::Vulkan
    glfw
)

option(NNPBR_BUILD_CODEC_TESTS "Build the independent BC6 decoder test oracle" OFF)
if(NNPBR_BUILD_CODEC_TESTS)
    add_library(bcdec_oracle SHARED tests/oracle/bcdec_wrapper.c)
endif()
