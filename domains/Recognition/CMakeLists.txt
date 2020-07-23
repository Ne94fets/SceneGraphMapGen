
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
		opencv_core
		opencv_tracking
		opencv_dnn_objdetect
		tensorflow_cc
		RecognitionDataTypes
		KinectDataTypes
)

set_target_properties(ObjectRecognition3d PROPERTIES
	CXX_STANDARD 17
	CXX_EXTENSIONS OFF)

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
		opencv_features2d
        pcl_common
        pcl_kdtree
		pcl_features
		pcl_filters
		pcl_registration
		pcl_sample_consensus
)

set_target_properties(PointCloudRegistration PROPERTIES
	CXX_STANDARD 17
	CXX_EXTENSIONS OFF)
