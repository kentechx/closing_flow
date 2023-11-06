cmake_minimum_required(VERSION 3.16)
project(py_closing_flow)

add_subdirectory(3rd_party/pybind11)
include_directories(3rd_party/libigl/include 3rd_party/eigen)
pybind11_add_module(py_closing_flow src/funcs.cpp)

set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "closing_flow/funcs")

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(py_closing_flow
                           PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})