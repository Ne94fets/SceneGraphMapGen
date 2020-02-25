##############################################################################

cmake_minimum_required(VERSION 2.8.4)

##############################################################################

project(GraphMap)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

##############################################################################

### set up download dir
set(DOWNLOAD_DIR "${CMAKE_SOURCE_DIR}/download")
if(NOT EXISTS "${DOWNLOAD_DIR}")
	message(STATUS "Creating download directory ${download_dir}")
	file(MAKE_DIRECTORY "${DOWNLOAD_DIR}")
	if(NOT EXISTS "${DOWNLOAD_DIR}")
		message(FATAL_ERROR "Failed to create download directory ${DOWNLOAD_DIR}")
	endif()
endif()

### install tensorflow
set(TF_TARGET_DIR "${CMAKE_SOURCE_DIR}/tensorflow")
if(NOT EXISTS "${TF_TARGET_DIR}")
	set(TF_DOWNLOAD_PATH "${DOWNLOAD_DIR}/tensorflow.tar.gz")
	if(NOT EXISTS "${TF_DOWNLOAD_PATH}")
		message(STATUS "Downloading tensorflow")
		file(DOWNLOAD
			"https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.15.0.tar.gz"
			"${TF_DOWNLOAD_PATH}"
			SHOW_PROGRESS)
	endif()
	file(MAKE_DIRECTORY "${TF_TARGET_DIR}")
	if(NOT EXISTS "${TF_TARGET_DIR}")
		message(FATAL_ERROR "Failed to create directory ${TF_TARGET_DIR}")
	endif()
	execute_process(
		COMMAND ${CMAKE_COMMAND} -E tar xzf "${TF_DOWNLOAD_PATH}"
		WORKING_DIRECTORY "${TF_TARGET_DIR}"
		RESULT_VARIABLE CMD_RES)
	if(NOT ${CMD_RES} EQUAL 0)
		message(FATAL_ERROR "Failed to extract ${TF_DOWNLOAD_PATH}")
	endif()
endif()

INCLUDE_DIRECTORIES("${TF_TARGET_DIR}/include")
LINK_DIRECTORIES("${TF_TARGET_DIR}/lib")

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
