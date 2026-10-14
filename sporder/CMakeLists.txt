cmake_minimum_required(VERSION 3.18)
project(sporder LANGUAGES CXX CUDA)

option(SPORDER_BUILD_PYTHON "Build the Python bindings" OFF)
option(SPORDER_BUILD_TESTS "Build the C API smoke test" OFF)

set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING
    "CUDA architectures (default: build machine's GPU)")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (default: Release)" FORCE)
endif()

find_package(CUDAToolkit REQUIRED)

file(GLOB_RECURSE SPORDER_SOURCES CONFIGURE_DEPENDS
    src/*.cpp
    src/*.cu
)

add_library(sporder
    ${SPORDER_SOURCES}
)

target_include_directories(sporder PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_include_directories(sporder PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_options(sporder PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:--ftz=true>
    $<$<COMPILE_LANGUAGE:CUDA>:--extra-device-vectorization>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

target_link_libraries(sporder PUBLIC
    CUDA::cudart
    CUDA::cusparse
)

if(SPORDER_BUILD_PYTHON)
    set_target_properties(sporder PROPERTIES POSITION_INDEPENDENT_CODE ON)

    find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)

    execute_process(
        COMMAND ${Python_EXECUTABLE} -m nanobind --cmake_dir
        OUTPUT_VARIABLE nanobind_ROOT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY
    )
    find_package(nanobind CONFIG REQUIRED)

    nanobind_add_module(_sporder python/bindings.cpp)
    target_link_libraries(_sporder PRIVATE sporder)

    install(TARGETS _sporder LIBRARY DESTINATION sporder)
endif()

if(SPORDER_BUILD_TESTS)
    # The smoke test is compiled as plain C on purpose: it guards the
    # header's C compatibility and the synchronous API contract.
    # Functional coverage lives in the Python suite under tests/.
    enable_language(C)
    enable_testing()

    add_executable(sporder_smoke tests/smoke.c)
    set_target_properties(sporder_smoke PROPERTIES C_STANDARD 99
                                                   C_STANDARD_REQUIRED ON)
    target_link_libraries(sporder_smoke PRIVATE sporder)

    add_test(NAME smoke COMMAND sporder_smoke)
endif()
