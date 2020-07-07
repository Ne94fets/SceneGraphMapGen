
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
		src/Detection.C

	LINK_LIBS
		MIRABase
		MIRAFramework
)
