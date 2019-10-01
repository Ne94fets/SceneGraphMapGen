##############################################################################

cmake_minimum_required(VERSION 2.8.4)

##############################################################################

project(GraphMap)

### needed by Qt
set(ENV{MIRA_PATH} "/opt/MIRA:/opt/MIRA-commercial:/localhome/demo/MIRA-fhws://localhome/demo/FHWSApplicationPython:/localhome/demo/fiwi-roboter_apps:/localhome/demo/demo:/localhome/demo/Software/fiwi-roboter_apps/fiwi-pathfinder:/localhome/demo/Kastner/GraphMap")

### libfreenect2
INCLUDE_DIRECTORIES("/localhome/demo/freenect2/include")
LINK_DIRECTORIES("/localhome/demo/freenect2/lib/")

##############################################################################

include(CMakeLists.mira)

##############################################################################

add_subdir(toolboxes)
add_subdir(domains)

##############################################################################
