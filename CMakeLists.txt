##############################################################################

cmake_minimum_required(VERSION 2.8.4)

##############################################################################

project(GraphMap)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


### libfreenect2
INCLUDE_DIRECTORIES("/home/steffen/Software/freenect2/include")
LINK_DIRECTORIES("/home/steffen/Software/freenect2/lib/")

##############################################################################

include(CMakeLists.mira)

##############################################################################

add_subdir(toolboxes)
add_subdir(domains)

##############################################################################
