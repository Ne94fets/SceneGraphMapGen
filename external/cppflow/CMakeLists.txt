project(cppflow)

include_directories("include")

add_library(cppflow SHARED src/Model.cpp src/Tensor.cpp)

