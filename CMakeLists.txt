##############################################################################

cmake_minimum_required(VERSION 2.8.4)

##############################################################################

project(GraphMap)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

##############################################################################

include(ExternalProject)

### set up download dir
set(DOWNLOAD_DIR "${CMAKE_SOURCE_DIR}/download")
if(NOT EXISTS "${DOWNLOAD_DIR}")
	message(STATUS "Creating download directory ${download_dir}")
	file(MAKE_DIRECTORY "${DOWNLOAD_DIR}")
	if(NOT EXISTS "${DOWNLOAD_DIR}")
		message(FATAL_ERROR "Failed to create download directory ${DOWNLOAD_DIR}")
	endif()
endif()

### install tensorflow python
execute_process(COMMAND pip install --user --upgrade tensorflow)

### create tensorflow model symlink
set(MODEL_SYMLINK_TARGET "${CMAKE_SOURCE_DIR}/external/models/faster_rcnn_inception_v2_coco_2018_01_28/saved_model/saved_model.pb")
set(MODEL_SYMLINK_DESTINATION "${CMAKE_SOURCE_DIR}/model.pb")
if(NOT EXISTS "${MODEL_SYMLINK_DESTINATION}")
	message(STATUS "Creating symlink to ${MODEL_SYMLINK_TARGET}")
	file(CREATE_LINK "${MODEL_SYMLINK_TARGET}" "${MODEL_SYMLINK_DESTINATION}")
endif()

### cppflow
add_subdirectory("external/cppflow")
INCLUDE_DIRECTORIES("external/cppflow/include")

### libfreenect2
INCLUDE_DIRECTORIES("/mnt/hdd/steffen/git/libfreenect2/include")
LINK_DIRECTORIES("/mnt/hdd/steffen/git/libfreenect2/lib/")

##############################################################################

include(CMakeLists.mira)

##############################################################################

add_subdir(toolboxes)
add_subdir(domains)

##############################################################################
