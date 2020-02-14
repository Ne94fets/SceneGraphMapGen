
MIRA_REQUIRE_PACKAGE(Kinect)

###############################################################################

MIRA_ADD_DOCUMENTATION(Kinect
	DIRS
		doc
		include
	DEPENDS
		#RobotDataTypes
)

###############################################################################

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_LIST_DIR}/include)

###############################################################################

MIRA_ADD_LIBRARY(KinectGrabber
	SHARED
	#PACKAGE $PACKAGE$
	SOURCE
		src/KinectGrabber.C

	LINK_LIBS
		MIRABase
		MIRAFramework
)

target_link_libraries(KinectGrabber libfreenect2.so)

###############################################################################

MIRA_ADD_LIBRARY(FloatProducer
	SHARED
	PACKAGE Kinect
	SOURCE
		src/FloatProducer.C

	LINK_LIBS
		MIRABase
		MIRAFramework
)

###############################################################################

MIRA_ADD_LIBRARY(FloatConsumer
	SHARED
	PACKAGE Kinect
	SOURCE
		src/FloatConsumer.C

	LINK_LIBS
		MIRABase
		MIRAFramework
)
