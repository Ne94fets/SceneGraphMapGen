cmake_minimum_required(VERSION 2.8)

add_library(tensorflow UNKNOWN IMPORTED)
set_property(TARGET tensorflow PROPERTY IMPORTED_LOCATION "../tensorflow/lib/tensorflow.so")
INCLUDE_DIRECTORIES("tensorflow/include")
LINK_DIRECTORIES("tensorflow/lib")

add_subdirectory("cppflow")
INCLUDE_DIRECTORIES("cppflow/include")

project(TensorTest)
add_executable(${PROJECT_NAME} "main.cpp")
target_link_libraries(${PROJECT_NAME} cppflow)
