
MIRA_REQUIRE_PACKAGE(Recognition)
MIRA_REQUIRE_PACKAGE(KinectDataTypes)
MIRA_REQUIRE_PACKAGE(RecognitionDataTypes)

###############################################################################

MIRA_ADD_DOCUMENTATION(Recognition
	DIRS
		doc
		include
	DEPENDS
        RecognitionDataTypes
)

###############################################################################

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_LIST_DIR}/include)

###############################################################################

MIRA_ADD_LIBRARY(ObjectRecognition3d
	SHARED
	PACKAGE Recognition
	SOURCE
		etc/ObjectRecognition3d.xml
        include/ObjectRecognition3d.h
		src/ObjectRecognition3d.C

	LINK_LIBS
		MIRABase
		MIRAFramework
        opencv_tracking
		libtensorflow_cc.so
        RecognitionDataTypes
        KinectDataTypes
)

###############################################################################

MIRA_ADD_LIBRARY(PointCloudRegistration
	SHARED
	PACKAGE Recognition
	SOURCE
		src/PointCloudRegistration.C

	LINK_LIBS
		MIRABase
		MIRAFramework
        KinectDataTypes
        pcl_common
        pcl_kdtree
        pcl_features
        pcl_registration
)
