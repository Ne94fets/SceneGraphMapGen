##############################################################################

cmake_minimum_required(VERSION 3.10)

##############################################################################

project(GraphMap)

##############################################################################

include(ProcessorCount)
ProcessorCount(NumCores)

### set up external dir
set(EXTERNAL_DIR "${CMAKE_SOURCE_DIR}/external")

### set up download dir
set(DOWNLOAD_DIR "${CMAKE_SOURCE_DIR}/download")
if(NOT EXISTS "${DOWNLOAD_DIR}")
	message(STATUS "Creating download directory ${download_dir}")
	file(MAKE_DIRECTORY "${DOWNLOAD_DIR}")
	if(NOT EXISTS "${DOWNLOAD_DIR}")
		message(FATAL_ERROR "Failed to create download directory ${DOWNLOAD_DIR}")
	endif()
endif()

### make sure opencv 4.3.0 is installed
set(OPENCV_DIR "${EXTERNAL_DIR}/opencv")
set(OPENCV_CONTRIB_DIR "${EXTERNAL_DIR}/opencv_contrib")
if(NOT EXISTS "${OPENCV_DIR}")
	### download opencv contrib
	set(OPENCV_CONTRIB_NAME "opencv_contrib-4.3.0")
	set(OPENCV_CONTRIB_DL "${DOWNLOAD_DIR}/${OPENCV_CONTRIB_NAME}.tar.gz")
	if(NOT EXISTS ${OPENCV_CONTRIB_DL})
		message(STATUS "Downloading ${OPENCV_NAME}")
		file(DOWNLOAD
			"https://github.com/opencv/opencv_contrib/archive/4.3.0.tar.gz"
			"${OPENCV_CONTRIB_DL}"
			SHOW_PROGRESS)
	endif()
	message(STATUS "Extracting ${OPENCV_CONTRIB_NAME}")
	execute_process(COMMAND
		${CMAKE_COMMAND} -E tar xf "${OPENCV_CONTRIB_DL}"
		WORKING_DIRECTORY "${DOWNLOAD_DIR}"
		ERROR_VARIABLE EXEC_ERR
		OUTPUT_VARIABLE EXEC_OUT)
	message(STATUS ${EXEC_OUT})
	if(${RESULT_VARIABLE})
		message(FATAL_ERROR ${EXEC_ERR})
	endif()
	message(STATUS "Moving opencv_contrib to external dir")
	file(RENAME "${DOWNLOAD_DIR}/${OPENCV_CONTRIB_NAME}" "${OPENCV_CONTRIB_DIR}")

	### download main opencv
	set(OPENCV_NAME "opencv-4.3.0")
	set(OPENCV_DL "${DOWNLOAD_DIR}/${OPENCV_NAME}.tar.gz")
	if(NOT EXISTS ${OPENCV_DL})
		message(STATUS "Downloading ${OPENCV_NAME}")
		file(DOWNLOAD
			"https://github.com/opencv/opencv/archive/4.3.0.tar.gz"
			"${OPENCV_DL}"
			SHOW_PROGRESS)
	endif()
	message(STATUS "Extracting ${OPENCV_NAME}")
	execute_process(COMMAND
		${CMAKE_COMMAND} -E tar xf "${OPENCV_DL}"
		WORKING_DIRECTORY "${DOWNLOAD_DIR}"
		ERROR_VARIABLE EXEC_ERR
		OUTPUT_VARIABLE EXEC_OUT)
	message(STATUS ${EXEC_OUT})
	if(${RESULT_VARIABLE})
		message(FATAL_ERROR ${EXEC_ERR})
	endif()

	### move opencv to external dir
	message(STATUS "Moving opencv to external dir")
	file(RENAME "${DOWNLOAD_DIR}/${OPENCV_NAME}" "${OPENCV_DIR}")
	file(MAKE_DIRECTORY "${OPENCV_DIR}/build")
	message(STATUS "Configure ${OPENCV_NAME}")
	execute_process(
		COMMAND cmake
			-DOPENCV_EXTRA_MODULES_PATH=${OPENCV_CONTRIB_DIR}/modules
			-DBUILD_OPENEXR=OFF
			-Dopencv_dnn_objdetect=ON
			-DBUILD_opencv_legacy=OFF
			-DCMAKE_BUILD_TYPE=Release
			-DCMAKE_INSTALL_PREFIX=${OPENCV_DIR} ..
		WORKING_DIRECTORY ${OPENCV_DIR}/build
		ERROR_VARIABLE EXEC_ERR)
	if(${RESULT_VARIABLE})
		message(FATAL_ERROR ${EXEC_ERR})
	endif()
	message(STATUS "Building ${OPENCV_NAME}")
	execute_process(
		COMMAND make -j ${NumCores}
		WORKING_DIRECTORY ${OPENCV_DIR}/build
		ERROR_VARIABLE EXEC_ERR)
	if(${RESULT_VARIABLE})
		message(FATAL_ERROR ${EXEC_ERR})
	endif()
	message(STATUS "Installing locally ${OPENCV_NAME}")
	execute_process(
		COMMAND make install
		WORKING_DIRECTORY ${OPENCV_DIR}/build
		ERROR_VARIABLE EXEC_ERR)
	if(${RESULT_VARIABLE})
		message(FATAL_ERROR ${EXEC_ERR})
	endif()
endif()

find_package(OpenCV 4.3.0 REQUIRED
	COMPONENTS core imgproc video dnn
	HINTS ${OPENCV_DIR})
include_directories(OpenCV_INCLUDE_DIR)

### create tensorflow model symlink
set(MODEL_SYMLINK_TARGET "${EXTERNAL_DIR}/models/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb")
set(MODEL_SYMLINK_DESTINATION "${CMAKE_SOURCE_DIR}/model.pb")
if(NOT EXISTS "${MODEL_SYMLINK_DESTINATION}")
	message(STATUS "Creating symlink to ${MODEL_SYMLINK_TARGET}")
	file(CREATE_LINK "${MODEL_SYMLINK_TARGET}" "${MODEL_SYMLINK_DESTINATION}")
endif()

### check for bazel
### use a tested bazel - tensorflow combination from
### https://www.tensorflow.org/install/source#tested_build_configurations
set(BAZEL_BIN_DIR "${EXTERNAL_DIR}/bazel")
set(BAZEL_EXEC "${BAZEL_BIN_DIR}/bazel")
if(NOT EXISTS "${BAZEL_EXEC}")
	set(BAZEL_DL "${DOWNLOAD_DIR}/bazel")
	if(NOT EXISTS ${BAZEL_DL})
        message(STATUS "Downloading bazel 2.0.0")
		file(DOWNLOAD
            "https://github.com/bazelbuild/bazel/releases/download/2.0.0/bazel-2.0.0-linux-x86_64"
			"${BAZEL_DL}"
			SHOW_PROGRESS)
	endif()
	file(MAKE_DIRECTORY ${BAZEL_BIN_DIR})
	file(COPY "${BAZEL_DL}"
		DESTINATION ${BAZEL_BIN_DIR}
		FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
	execute_process(
		COMMAND ${BAZEL_EXEC} --help
		OUTPUT_VARIABLE EXEC_OUT)
	message(STATUS ${EXEC_OUT})
endif()

### check for tensorflow
set(TF_DIR "${EXTERNAL_DIR}/tensorflow")
if(NOT EXISTS "${TF_DIR}")
    message(STATUS "Downloading Tensorflow")
	execute_process(
		COMMAND git clone "https://github.com/tensorflow/tensorflow.git" ${TF_DIR}
		ERROR_VARIABLE EXEC_ERR)
	if(${RESULT_VARIABLE})
		message(FATAL_ERROR ${EXEC_ERR})
	endif()
    message(STATUS "Check out r2.2")
    execute_process(
        COMMAND git checkout r2.2
		WORKING_DIRECTORY ${TF_DIR}
		ERROR_VARIABLE EXEC_ERR)
	if(${RESULT_VARIABLE})
		message(FATAL_ERROR ${EXEC_ERR})
	endif()
#    message(STATUS "Run ./configure")
#    execute_process(
#        COMMAND ${CMAKE_COMMAND} -E env PATH="${BAZEL_BIN_DIR}:$ENV{PATH}"
#            bash echo "\n\n\n\n\n\n\n\n" | ./configure
#        WORKING_DIRECTORY ${TF_DIR}
#        ERROR_VARIABLE EXEC_ERR
#        OUTPUT_VARIABLE EXEC_OUT)
#    message(STATUS "${EXEC_OUT}")
#    if(${RESULT_VARIABLE})
#        message(FATAL_ERROR ${EXEC_ERR})
#    endif()
    message(STATUS "Building //tensorflow:libtensorflow_cc.so")
    execute_process(
        COMMAND
            ${BAZEL_EXEC} build
                --config=xla
                --copt=-march=native
                --copt=-Wno-sign-compare
                --host_copt=-march=native
                --define with_default_optimizations=true
                --copt=-O3
                --action_env TF_CONFIGURE_IOS="0"
                //tensorflow:libtensorflow_cc.so
		WORKING_DIRECTORY ${TF_DIR}
		ERROR_VARIABLE EXEC_ERR)
	if(${RESULT_VARIABLE})
		message(FATAL_ERROR ${EXEC_ERR})
	endif()
endif()

INCLUDE_DIRECTORIES("${TF_DIR}/")
INCLUDE_DIRECTORIES("${TF_DIR}/bazel-out/host/bin")
INCLUDE_DIRECTORIES("${TF_DIR}/bazel-tensorflow/external/eigen_archive")
INCLUDE_DIRECTORIES("${TF_DIR}/bazel-tensorflow/external/com_google_absl")
LINK_DIRECTORIES("${TF_DIR}/bazel-bin/tensorflow")

### download protobuf 3.8.0
set(PROTOBUF_DIR "${EXTERNAL_DIR}/protobuf")
if(NOT EXISTS "${PROTOBUF_DIR}")
	message(STATUS "Downloading protobuf")
	execute_process(
        COMMAND git clone "https://github.com/protocolbuffers/protobuf.git" ${PROTOBUF_DIR}
        ERROR_VARIABLE EXEC_ERR)
    if(${RESULT_VARIABLE})
        message(FATAL_ERROR ${EXEC_ERR})
    endif()
    message(STATUS "Check out 3.8.0")
    execute_process(
        COMMAND git checkout v3.8.0
        WORKING_DIRECTORY ${PROTOBUF_DIR}
        ERROR_VARIABLE EXEC_ERR)
    if(${RESULT_VARIABLE})
        message(FATAL_ERROR ${EXEC_ERR})
    endif()
	message(STATUS "Init submodules")
	execute_process(
		COMMAND git submodule update --init --recursive
        WORKING_DIRECTORY ${PROTOBUF_DIR}
        ERROR_VARIABLE EXEC_ERR)
    if(${RESULT_VARIABLE})
        message(FATAL_ERROR ${EXEC_ERR})
    endif()
	message(STATUS "Running autogen.sh")
	execute_process(
		COMMAND ./autogen.sh
        WORKING_DIRECTORY ${PROTOBUF_DIR}
        ERROR_VARIABLE EXEC_ERR)
    if(${RESULT_VARIABLE})
        message(FATAL_ERROR ${EXEC_ERR})
    endif()
	message(STATUS "Running configure")
	execute_process(
		COMMAND ./configure --prefix=${PROTOBUF_DIR}
        WORKING_DIRECTORY ${PROTOBUF_DIR}
        ERROR_VARIABLE EXEC_ERR)
    if(${RESULT_VARIABLE})
        message(FATAL_ERROR ${EXEC_ERR})
    endif()
    message(STATUS "Building potobuf with make -j ${NumCores} -pipe")
	execute_process(
		COMMAND make -j ${NumCores} -pipe
        WORKING_DIRECTORY ${PROTOBUF_DIR}
        ERROR_VARIABLE EXEC_ERR)
    if(${RESULT_VARIABLE})
        message(FATAL_ERROR ${EXEC_ERR})
    endif()
	message(STATUS "Checking build")
	execute_process(
		COMMAND make check -j ${NumCores} -pipe
        WORKING_DIRECTORY ${PROTOBUF_DIR}
        ERROR_VARIABLE EXEC_ERR)
    if(${RESULT_VARIABLE})
        message(FATAL_ERROR ${EXEC_ERR})
    endif()
	message(STATUS "Installing protobuf")
	execute_process(
		COMMAND make install
        WORKING_DIRECTORY ${PROTOBUF_DIR}
        ERROR_VARIABLE EXEC_ERR)
    if(${RESULT_VARIABLE})
        message(FATAL_ERROR ${EXEC_ERR})
    endif()
endif()
INCLUDE_DIRECTORIES("${PROTOBUF_DIR}/include")
LINK_DIRECTORIES("${PROTOBUF_DIR}/lib")

### download neo4j
set(NEO4J_DIR "${EXTERNAL_DIR}/neo4j")
if(NOT EXISTS "${NEO4J_DIR}")
    set(NEO4J_NAME "neo4j-community-3.5.17")
    set(NEO4J_DL "${DOWNLOAD_DIR}/${NEO4J_NAME}-unix.tar.gz")
    if(NOT EXISTS ${NEO4J_DL})
        message(STATUS "Downloading ${NEO4J_NAME}")
        file(DOWNLOAD
            "https://neo4j.com/artifact.php?name=neo4j-community-3.5.17-unix.tar.gz"
            "${NEO4J_DL}"
            SHOW_PROGRESS)
    endif()
    message(STATUS "Extracting ${NEO4J_NAME}")
    execute_process(COMMAND
        ${CMAKE_COMMAND} -E tar xf "${NEO4J_DL}"
        WORKING_DIRECTORY "${DOWNLOAD_DIR}"
        ERROR_VARIABLE EXEC_ERR
        OUTPUT_VARIABLE EXEC_OUT)
    message(STATUS ${EXEC_OUT})
    if(${RESULT_VARIABLE})
        message(FATAL_ERROR ${EXEC_ERR})
	endif()
    message(STATUS "Moving to external dir")
	file(RENAME "${DOWNLOAD_DIR}/${NEO4J_NAME}" "${NEO4J_DIR}")
endif()

### download libneo4j-client
set(LIBNEO4J_DIR "${EXTERNAL_DIR}/libneo4j-client")
if(NOT EXISTS "${LIBNEO4J_DIR}")
    set(LIBNEO4J_NAME "libneo4j-client-master")
    set(LIBNEO4J_DL "${DOWNLOAD_DIR}/${LIBNEO4J_NAME}.tar.gz")
    if(NOT EXISTS ${LIBNEO4J_DL})
        message(STATUS "Downloading ${LIBNEO4J_NAME}")
        file(DOWNLOAD
            "https://github.com/cleishm/libneo4j-client/archive/master.zip"
            "${LIBNEO4J_DL}"
            SHOW_PROGRESS)
    endif()
    message(STATUS "Extracting ${LIBNEO4J_NAME}")
    execute_process(COMMAND
        ${CMAKE_COMMAND} -E tar xf "${LIBNEO4J_DL}"
        WORKING_DIRECTORY "${DOWNLOAD_DIR}"
        ERROR_VARIABLE EXEC_ERR
        OUTPUT_VARIABLE EXEC_OUT)
    message(STATUS ${EXEC_OUT})
    if(${RESULT_VARIABLE})
        message(FATAL_ERROR ${EXEC_ERR})
	endif()
    message(STATUS "Moving to external dir")
    file(RENAME "${DOWNLOAD_DIR}/${LIBNEO4J_NAME}" "${LIBNEO4J_DIR}")
    message(STATUS "Autogen ${LIBNEO4J_NAME}")
    execute_process(
        COMMAND ./autogen.sh
        WORKING_DIRECTORY ${LIBNEO4J_DIR}
        ERROR_VARIABLE EXEC_ERR)
    if(${RESULT_VARIABLE})
        message(FATAL_ERROR ${EXEC_ERR})
    endif()
    message(STATUS "Configure ${LIBNEO4J_NAME}")
    execute_process(
        COMMAND ./configure --disable-tools
        WORKING_DIRECTORY ${LIBNEO4J_DIR}
        ERROR_VARIABLE EXEC_ERR)
    if(${RESULT_VARIABLE})
        message(FATAL_ERROR ${EXEC_ERR})
    endif()
    message(STATUS "Cleaning than building ${LIBNEO4J_NAME}")
    execute_process(
        COMMAND make clean check -j ${NumCores}
        WORKING_DIRECTORY ${LIBNEO4J_DIR}
        ERROR_VARIABLE EXEC_ERR)
    if(${RESULT_VARIABLE})
        message(FATAL_ERROR ${EXEC_ERR})
    endif()
    message(STATUS "Installing locally ${LIBNEO4J_NAME}")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E env DESTDIR=${LIBNEO4J_DIR} make install
        WORKING_DIRECTORY ${LIBNEO4J_DIR}
        ERROR_VARIABLE EXEC_ERR)
    if(${RESULT_VARIABLE})
        message(FATAL_ERROR ${EXEC_ERR})
    endif()
endif()

INCLUDE_DIRECTORIES("${LIBNEO4J_DIR}/usr/local/include")
LINK_DIRECTORIES("${LIBNEO4J_DIR}/usr/local/lib")

### download pcl
set(PCL_DIR "${EXTERNAL_DIR}/pcl")
if(NOT EXISTS "${PCL_DIR}")
    set(PCL_NAME "pcl-pcl-1.11.0")
    set(PCL_DL "${DOWNLOAD_DIR}/${PCL_NAME}.tar.gz")
    if(NOT EXISTS ${PCL_DL})
        message(STATUS "Downloading ${PCL_NAME}")
        file(DOWNLOAD
            "https://github.com/PointCloudLibrary/pcl/archive/pcl-1.11.0.tar.gz"
            "${PCL_DL}"
            SHOW_PROGRESS)
    endif()
    message(STATUS "Extracting ${PCL_NAME}")
    execute_process(COMMAND
        ${CMAKE_COMMAND} -E tar xf "${PCL_DL}"
        WORKING_DIRECTORY "${DOWNLOAD_DIR}"
        ERROR_VARIABLE EXEC_ERR
        OUTPUT_VARIABLE EXEC_OUT)
    message(STATUS ${EXEC_OUT})
    if(${RESULT_VARIABLE})
        message(FATAL_ERROR ${EXEC_ERR})
	endif()
    message(STATUS "Moving to external dir")
	file(RENAME "${DOWNLOAD_DIR}/${PCL_NAME}" "${PCL_DIR}")
    file(MAKE_DIRECTORY "${PCL_DIR}/build")
    message(STATUS "Configure ${PCL_NAME}")
    execute_process(
		COMMAND cmake -DCMAKE_BUILD_TYPE=Release ..
        WORKING_DIRECTORY ${PCL_DIR}/build
        ERROR_VARIABLE EXEC_ERR)
    if(${RESULT_VARIABLE})
        message(FATAL_ERROR ${EXEC_ERR})
    endif()
    message(STATUS "Building ${PCL_NAME}")
    execute_process(
        COMMAND make -j ${NumCores}
        WORKING_DIRECTORY ${PCL_DIR}/build
        ERROR_VARIABLE EXEC_ERR)
    if(${RESULT_VARIABLE})
        message(FATAL_ERROR ${EXEC_ERR})
    endif()
    message(STATUS "Installing locally ${PCL_NAME}")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E env DESTDIR=${PCL_DIR} make install
        WORKING_DIRECTORY ${PCL_DIR}
        ERROR_VARIABLE EXEC_ERR)
    if(${RESULT_VARIABLE})
        message(FATAL_ERROR ${EXEC_ERR})
    endif()
endif()

INCLUDE_DIRECTORIES("${PCL_DIR}/usr/local/include/pcl-1.11")
LINK_DIRECTORIES("${PCL_DIR}/usr/local/lib")
find_package(VTK REQUIRED)
INCLUDE_DIRECTORIES("/usr/include/vtk-8.2")

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${PCL_DIR}/usr/local/share/pcl-1.11/Modules")
find_package("FLANN" REQUIRED)

### libfreenect2
set(LIBFREENECT2_DIR "${EXTERNAL_DIR}/libfreenect2")
if(NOT EXISTS "${LIBFREENECT2_DIR}")
    set(LIBFREENECT2_NAME "libfreenect2-fd64c5d9b214df6f6a55b4419357e51083f15d93")
    set(LIBFREENECT2_DL "${DOWNLOAD_DIR}/fd64c5d9b214df6f6a55b4419357e51083f15d93.zip")
    if(NOT EXISTS ${LIBFREENECT2_DL})
        message(STATUS "Downloading ${LIBFREENECT2_NAME}")
        file(DOWNLOAD
            "https://github.com/OpenKinect/libfreenect2/archive/fd64c5d9b214df6f6a55b4419357e51083f15d93.zip"
            "${LIBFREENECT2_DL}"
            SHOW_PROGRESS)
    endif()
    message(STATUS "Extracting ${LIBFREENECT2_NAME}")
    execute_process(COMMAND
        ${CMAKE_COMMAND} -E tar xf "${LIBFREENECT2_DL}"
        WORKING_DIRECTORY "${DOWNLOAD_DIR}"
        ERROR_VARIABLE EXEC_ERR
        OUTPUT_VARIABLE EXEC_OUT)
    message(STATUS ${EXEC_OUT})
    if(${RESULT_VARIABLE})
        message(FATAL_ERROR ${EXEC_ERR})
    endif()
    message(STATUS "Moving to external dir")
    file(RENAME "${DOWNLOAD_DIR}/${LIBFREENECT2_NAME}" "${LIBFREENECT2_DIR}")
    file(MAKE_DIRECTORY "${LIBFREENECT2_DIR}/build")
    message(STATUS "Configure ${LIBFREENECT2_NAME}")
    execute_process(
        COMMAND cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${LIBFREENECT2_DIR} ..
        WORKING_DIRECTORY ${LIBFREENECT2_DIR}/build
		ERROR_VARIABLE EXEC_ERR
		OUTPUT_VARIABLE EXEC_OUT)
	message(STATUS ${EXEC_OUT})
    if(${RESULT_VARIABLE})
        message(FATAL_ERROR ${EXEC_ERR})
    endif()
    message(STATUS "Building ${LIBFREENECT2_NAME}")
    execute_process(
        COMMAND make -j ${NumCores}
        WORKING_DIRECTORY ${LIBFREENECT2_DIR}/build
        ERROR_VARIABLE EXEC_ERR)
    if(${RESULT_VARIABLE})
        message(FATAL_ERROR ${EXEC_ERR})
    endif()
    message(STATUS "Installing locally ${LIBFREENECT2_NAME}")
    execute_process(
        COMMAND make install
        WORKING_DIRECTORY ${LIBFREENECT2_DIR}/build
        ERROR_VARIABLE EXEC_ERR)
    if(${RESULT_VARIABLE})
        message(FATAL_ERROR ${EXEC_ERR})
    endif()
endif()

INCLUDE_DIRECTORIES("${LIBFREENECT2_DIR}/include")
LINK_DIRECTORIES("${LIBFREENECT2_DIR}/lib")

##############################################################################

include(CMakeLists.mira)

##############################################################################

add_subdir(toolboxes)
add_subdir(domains)

##############################################################################
