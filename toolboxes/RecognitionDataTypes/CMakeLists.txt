
MIRA_REQUIRE_PACKAGE(RecognitionDataTypes)

###############################################################################

MIRA_ADD_DOCUMENTATION(RecognitionDataTypes
	DIRS
		doc
		include
	DEPENDS
		#RobotDataTypes
)

###############################################################################

MIRA_ADD_LIBRARY(RecognitionDataTypes
	SHARED
	PACKAGE RecognitionDataTypes
	SOURCE
		include/recognitiondatatypes/RecognitionDataTypesExports.h
		include/recognitiondatatypes/Detection.h
		include/recognitiondatatypes/TrackerGenerator.h
		src/Detection.C
		src/TrackerGenerator.C

	LINK_LIBS
		MIRABase
		MIRAFramework
		opencv_core
		opencv_features2d
		opencv_objdetect
		opencv_highgui
		opencv_tracking
)
