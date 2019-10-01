
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
