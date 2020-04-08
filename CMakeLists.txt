##############################################################################

cmake_minimum_required(VERSION 2.8.4)

##############################################################################

project(GraphMap)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

##############################################################################

include(ProcessorCount)
ProcessorCount(NumCores)

### set up download dir
set(DOWNLOAD_DIR "${CMAKE_SOURCE_DIR}/download")
if(NOT EXISTS "${DOWNLOAD_DIR}")
	message(STATUS "Creating download directory ${download_dir}")
	file(MAKE_DIRECTORY "${DOWNLOAD_DIR}")
	if(NOT EXISTS "${DOWNLOAD_DIR}")
		message(FATAL_ERROR "Failed to create download directory ${DOWNLOAD_DIR}")
	endif()
endif()

### create tensorflow model symlink
set(MODEL_SYMLINK_TARGET "${CMAKE_SOURCE_DIR}/external/models/faster_rcnn_inception_v2_coco_2018_01_28/saved_model/saved_model.pb")
set(MODEL_SYMLINK_DESTINATION "${CMAKE_SOURCE_DIR}/model.pb")
if(NOT EXISTS "${MODEL_SYMLINK_DESTINATION}")
	message(STATUS "Creating symlink to ${MODEL_SYMLINK_TARGET}")
	file(CREATE_LINK "${MODEL_SYMLINK_TARGET}" "${MODEL_SYMLINK_DESTINATION}")
endif()

### check for bazel
set(BAZEL_DIR "${CMAKE_SOURCE_DIR}/external/bazel")
set(BAZEL_BIN_DIR "${BAZEL_DIR}/bazel-bin/src/")
if(NOT EXISTS "${BAZEL_DIR}")
	message(STATUS "Installing bazel")
	execute_process(
		COMMAND git clone "https://github.com/bazelbuild/bazel.git" ${BAZEL_DIR})
	execute_process(
		COMMAND git checkout 2.2.0
		COMMAND bazel build //src:bazel-dev
		COMMAND ln -s ${BAZEL_BIN_DIR}/bazel-dev ${BAZEL_BIN_DIR}/bazel
		WORKING_DIRECTORY ${BAZEL_DIR})
endif()

### check for tensorflow
set(TF_DIR "${CMAKE_SOURCE_DIR}/external/tensorflow")
if(NOT EXISTS "${TF_DIR}")
	message(STATUS "Installing Tensorflow")
	execute_process(
		COMMAND git clone "https://github.com/tensorflow/tensorflow.git" ${TF_DIR})
	execute_process(
		COMMAND git checkout r2.1
		COMMAND ${CMAKE_COMMAND} -E env PATH=${BAZEL_BIN_DIR}:$ENV{PATH} ./configure
		COMMAND ${CMAKE_COMMAND} -E env PATH=${BAZEL_BIN_DIR}:$ENV{PATH} bazel build //tensorflow:libtensorflow_cc.so
		WORKING_DIRECTORY ${TF_DIR})
endif()

INCLUDE_DIRECTORIES("${TF_DIR}/")
INCLUDE_DIRECTORIES("${TF_DIR}/bazel-out/host/bin")
INCLUDE_DIRECTORIES("${TF_DIR}/bazel-tensorflow/external/eigen_archive")
LINK_DIRECTORIES("${TF_DIR}/bazel-bin/tensorflow")

### download protobuf 3.8.0
set(PROTOBUF_DIR "${CMAKE_SOURCE_DIR}/external/protobuf")
if(NOT EXISTS "${PROTOBUF_DIR}")
	message(STATUS "Downloading protobuf")
	execute_process(
		COMMAND git clone "https://github.com/protocolbuffers/protobuf.git" ${PROTOBUF_DIR})
	message(STATUS "Check out 3.8.0")
	execute_process(
		COMMAND git checkout v3.8.0
		WORKING_DIRECTORY ${PROTOBUF_DIR})
	message(STATUS "Init submodules")
	execute_process(
		COMMAND git submodule update --init --recursive
		WORKING_DIRECTORY ${PROTOBUF_DIR})
	message(STATUS "Running autogen.sh")
	execute_process(
		COMMAND ./autogen.sh
		WORKING_DIRECTORY ${PROTOBUF_DIR})
	message(STATUS "Running configure")
	execute_process(
		COMMAND ./configure --prefix=${PROTOBUF_DIR}
		WORKING_DIRECTORY ${PROTOBUF_DIR})
	message(STATUS "Building potobuf")
	execute_process(
		COMMAND make -j ${NumCores} -pipe
		WORKING_DIRECTORY ${PROTOBUF_DIR})
	message(STATUS "Checking build")
	execute_process(
		COMMAND make check -j ${NumCores} -pipe
		WORKING_DIRECTORY ${PROTOBUF_DIR})
	message(STATUS "Installing protobuf")
	execute_process(
		COMMAND make install
		WORKING_DIRECTORY ${PROTOBUF_DIR})
endif()
INCLUDE_DIRECTORIES("${PROTOBUF_DIR}/include")
LINK_DIRECTORIES("${PROTOBUF_DIR}/lib")

### download abseil (absl)
set(BUILD_TESTING OFF)
set(ABSL_DIR "${CMAKE_SOURCE_DIR}/external/absl")
if(NOT EXISTS "${ABSL_DIR}")
	message(STATUS "Downloading abseil")
	execute_process(
		COMMAND git clone "https://github.com/abseil/abseil-cpp.git" ${ABSL_DIR})
	message(STATUS "Check out 20200225.1")
	execute_process(
		COMMAND git checkout 20200225.1
		WORKING_DIRECTORY ${ABSL_DIR})
endif()
add_subdirectory(${ABSL_DIR})
INCLUDE_DIRECTORIES("${ABSL_DIR}")

### libfreenect2
INCLUDE_DIRECTORIES("/mnt/hdd/steffen/git/libfreenect2/include")
LINK_DIRECTORIES("/mnt/hdd/steffen/git/libfreenect2/lib/")

##############################################################################

include(CMakeLists.mira)

##############################################################################

add_subdir(toolboxes)
add_subdir(domains)

##############################################################################
