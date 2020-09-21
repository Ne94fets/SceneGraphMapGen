# Generating a Graph-Based Scene Description with a Mobile RGBD-Camera

This repository contains the comple code of an end-to-end pipeline used to generate the graph-based scene description. A single, moblie RGBD-camera is used to capture the scene. The software runs on a SCITOS G5 from MetraLabs at either 10Hz or 30Hz. The robot is equipped with an eight core Intel Core i7-6700T CPU running at 2.8GHz and 8GHz RAM.

## Hardware Requirements
- RGBD-camera (Code is based on libfreenect2 and uses a Microsoft Kinect v2)

## Dependencies
Software needed to build:
- non-comercial MIRA-framework
- python (required to run evaluation scripts)

Software downloaded by the cmake script:
- OpenCV 4.3.0 + contrib
- bazel 2.0.0
- tensorflow r2.2
- protobuf 3.8.0
- neo4j (server executable, java required)
- libneo4j-client
- PCL
- libfreenect2
- rapidjson (for evaluation only)

## How to build
Run
´´´
git clone https://github.com/Ne94fets/SceneGraphMapGen.git
mkdir build
cd build
cmake ../
make
´´´

Instead of running cmake you can also import the CMakeLists.txt in your favourite Editor, like qtcreator.

## How to Evaluate the Camera Pose Estimation
see [here](external/evalData/PUTK/README.md)

## How to Evaluate Object Detection and Object Position Estimation
see [here](external/evalData/SUNRGBD-EvalScripts/README.md)
