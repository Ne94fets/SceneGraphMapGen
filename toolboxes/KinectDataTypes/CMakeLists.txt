
MIRA_REQUIRE_PACKAGE(KinectDataTypes)

###############################################################################

MIRA_ADD_DOCUMENTATION(KinectDataTypes
	DIRS
		doc
		include
	DEPENDS
		#RobotDataTypes
)

###############################################################################

MIRA_ADD_LIBRARY(KinectDataTypes
	SHARED
	PACKAGE KinectDataTypes
	SOURCE
		include/kinectdatatypes/KinectDataTypesExports.h
		include/kinectdatatypes/Types.h
		src/Types.C

	LINK_LIBS
		MIRABase
		MIRAFramework
		libfreenect2.so
)
