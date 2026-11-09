cmake_minimum_required(VERSION 3.16)
project(NNPBRPluginChecks LANGUAGES CXX)
add_executable(nnpbr_format_test format_test.cpp)
target_compile_features(nnpbr_format_test PRIVATE cxx_std_17)
target_include_directories(nnpbr_format_test PRIVATE ../NNPBRRuntime/Source/NNPBRRuntime/Public)
enable_testing()
foreach(model IN ITEMS red-plaid-64 hammered-copper-512 red-plaid-2k)
    add_test(NAME ${model} COMMAND nnpbr_format_test "${CMAKE_CURRENT_SOURCE_DIR}/../../models/${model}/export")
endforeach()
find_program(GLSLC glslc REQUIRED)
find_program(SPIRV_VAL spirv-val REQUIRED)
set(PROBE_SPV "${CMAKE_CURRENT_BINARY_DIR}/shader_check.spv")
add_custom_command(OUTPUT "${PROBE_SPV}"
    COMMAND "${GLSLC}" -x hlsl -fshader-stage=compute -O
            "-I${CMAKE_CURRENT_SOURCE_DIR}/../NNPBRRuntime/Shaders/Private"
            "${CMAKE_CURRENT_SOURCE_DIR}/shader_check.hlsl" -o "${PROBE_SPV}"
    DEPENDS shader_check.hlsl ../NNPBRRuntime/Shaders/Private/NNPBRMaterial.ush VERBATIM)
add_custom_target(nnpbr_shader_check ALL DEPENDS "${PROBE_SPV}")
add_test(NAME decoder_shader COMMAND "${SPIRV_VAL}" "${PROBE_SPV}")

find_package(Vulkan REQUIRED)
add_executable(nnpbr_shader_gpu_test shader_gpu_test.cpp)
target_compile_features(nnpbr_shader_gpu_test PRIVATE cxx_std_17)
target_include_directories(nnpbr_shader_gpu_test PRIVATE ../NNPBRRuntime/Source/NNPBRRuntime/Public)
target_link_libraries(nnpbr_shader_gpu_test PRIVATE Vulkan::Vulkan)
add_dependencies(nnpbr_shader_gpu_test nnpbr_shader_check)
foreach(model IN ITEMS red-plaid-64 hammered-copper-512 red-plaid-2k)
    add_test(NAME shader_gpu_${model} COMMAND nnpbr_shader_gpu_test "${PROBE_SPV}"
             "${CMAKE_CURRENT_SOURCE_DIR}/../../models/${model}/export/decoder_fp16.bin")
    set_tests_properties(shader_gpu_${model} PROPERTIES SKIP_RETURN_CODE 77)
endforeach()
