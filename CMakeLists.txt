cmake_minimum_required(VERSION 3.20)
project(helmholtz-wods LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)

# --- GLSL → SPIR-V -----------------------------------------------------------
find_program(GLSLANG_VALIDATOR glslangValidator REQUIRED)
set(SHADER_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(SHADER_SPV_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders/spv)
file(MAKE_DIRECTORY ${SHADER_SPV_DIR})

set(SHADER_SOURCES
  fill.comp
  pass_a.comp
  pass_c.comp
  pass_d.vert
  pass_d.frag
)
set(SPV_OUTPUTS)
foreach(SHADER ${SHADER_SOURCES})
  set(SRC ${SHADER_SRC_DIR}/${SHADER})
  set(SPV ${SHADER_SPV_DIR}/${SHADER}.spv)
  add_custom_command(
    OUTPUT ${SPV}
    COMMAND ${GLSLANG_VALIDATOR} -V -o ${SPV} ${SRC}
    DEPENDS ${SRC}
    COMMENT "SPIR-V ${SHADER}"
    VERBATIM
  )
  list(APPEND SPV_OUTPUTS ${SPV})
endforeach()
add_custom_target(shaders ALL DEPENDS ${SPV_OUTPUTS})

# --- Core library ------------------------------------------------------------
add_library(hwods_core
  src/algo/sdf.cpp
  src/algo/tiles.cpp
  src/algo/weight.cpp
  src/algo/walk_on_stars.cpp
  src/eval/laplace.cpp
  src/eval/fdm.cpp
  src/eval/rmse.cpp
)
target_include_directories(hwods_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
# libm is part of libSystem on Apple; -lm is harmless elsewhere.
if(NOT APPLE)
  target_link_libraries(hwods_core PUBLIC m)
endif()

# --- GPU (GLFW + Vulkan compute fill) ----------------------------------------
add_library(hwods_gpu
  src/gpu/vulkan_fill.cpp
)
target_include_directories(hwods_gpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(hwods_gpu PUBLIC Vulkan::Vulkan glfw)
target_compile_definitions(hwods_gpu PUBLIC
  HWODS_SHADER_DIR="${SHADER_SPV_DIR}"
)
add_dependencies(hwods_gpu shaders)

# --- Apps --------------------------------------------------------------------
add_executable(helmholtz-wods src/main.cpp)
target_link_libraries(helmholtz-wods PRIVATE hwods_gpu hwods_core)

add_executable(laplace-check scripts/laplace_check.cpp)
target_link_libraries(laplace-check PRIVATE hwods_core)

add_executable(hwods_tests tests/algo_tests.cpp)
target_link_libraries(hwods_tests PRIVATE hwods_core)
enable_testing()
add_test(NAME algo_tests COMMAND hwods_tests)

install(TARGETS helmholtz-wods laplace-check RUNTIME DESTINATION bin)
