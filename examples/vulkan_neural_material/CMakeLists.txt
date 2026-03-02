cmake_minimum_required(VERSION 3.24)
project(vulkan_neural_material LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)

if(NOT DEFINED ENV{VULKAN_SDK})
  message(FATAL_ERROR "VULKAN_SDK env var is required to locate dxc.")
endif()

set(VK_SDK_BIN "$ENV{VULKAN_SDK}/Bin")
find_program(DXC_EXECUTABLE NAMES dxc HINTS "${VK_SDK_BIN}" REQUIRED)

include(FetchContent)

FetchContent_Declare(
  glfw
  GIT_REPOSITORY https://github.com/glfw/glfw.git
  GIT_TAG 3.4
)
FetchContent_MakeAvailable(glfw)

FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

FetchContent_Declare(
  imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG v1.91.8
)
FetchContent_MakeAvailable(imgui)

add_library(imgui_vulkan STATIC
  ${imgui_SOURCE_DIR}/imgui.cpp
  ${imgui_SOURCE_DIR}/imgui_draw.cpp
  ${imgui_SOURCE_DIR}/imgui_tables.cpp
  ${imgui_SOURCE_DIR}/imgui_widgets.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
)
target_include_directories(imgui_vulkan PUBLIC
  ${imgui_SOURCE_DIR}
  ${imgui_SOURCE_DIR}/backends
)
target_link_libraries(imgui_vulkan PUBLIC glfw Vulkan::Vulkan)

set(SHADER_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(SHADER_BIN_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_BIN_DIR})

set(VS_SPV ${SHADER_BIN_DIR}/fullscreen_vs.spv)
set(FS_SPV ${SHADER_BIN_DIR}/neural_material_fs.spv)

add_custom_command(
  OUTPUT ${VS_SPV}
  COMMAND ${DXC_EXECUTABLE} -spirv -fspv-target-env=vulkan1.3 -T vs_6_0 -E main -Fo ${VS_SPV} ${SHADER_SRC_DIR}/fullscreen_vs.hlsl
  DEPENDS ${SHADER_SRC_DIR}/fullscreen_vs.hlsl
  COMMENT "Compiling fullscreen_vs.hlsl"
)

add_custom_command(
  OUTPUT ${FS_SPV}
  COMMAND ${DXC_EXECUTABLE} -spirv -fspv-target-env=vulkan1.3 -T ps_6_6 -E main -Fo ${FS_SPV} ${SHADER_SRC_DIR}/neural_material_fs.hlsl
  DEPENDS ${SHADER_SRC_DIR}/neural_material_fs.hlsl
  COMMENT "Compiling neural_material_fs.hlsl"
)

add_custom_target(compile_shaders ALL DEPENDS ${VS_SPV} ${FS_SPV})

add_executable(vulkan_neural_material src/main.cpp)
add_dependencies(vulkan_neural_material compile_shaders)

target_include_directories(vulkan_neural_material PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_definitions(vulkan_neural_material PRIVATE
  SHADER_BIN_DIR="${SHADER_BIN_DIR}"
)

target_link_libraries(vulkan_neural_material PRIVATE
  Vulkan::Vulkan
  glfw
  imgui_vulkan
  nlohmann_json::nlohmann_json
)

if(MSVC)
  target_compile_options(vulkan_neural_material PRIVATE /W4)
else()
  target_compile_options(vulkan_neural_material PRIVATE -Wall -Wextra -Wpedantic)
endif()
