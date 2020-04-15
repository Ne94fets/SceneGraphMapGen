
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
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/toolboxes/KinectDataTypes/include)

###############################################################################

MIRA_ADD_LIBRARY(KinectGrabber
	SHARED
	PACKAGE Kinect
	SOURCE
        include/KinectGrabber.h
		src/KinectGrabber.C

	LINK_LIBS
		MIRABase
        MIRAFramework
        libfreenect2.so
        KinectDataTypes
)

###############################################################################

MIRA_ADD_LIBRARY(FloatProducer
	SHARED
	PACKAGE Kinect
	SOURCE
		src/FloatProducer.C
        etc/FloatProducer.xml

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
        etc/FloatConsumer.xml

	LINK_LIBS
		MIRABase
		MIRAFramework
)

###############################################################################

MIRA_ADD_LIBRARY(DummyKinectSubscriber
	SHARED
	PACKAGE Kinect
	SOURCE
		src/DummyKinectSubscriber.C

	LINK_LIBS
		MIRABase
		MIRAFramework
)
