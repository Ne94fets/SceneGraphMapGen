
MIRA_REQUIRE_PACKAGE(Evaluation)
MIRA_REQUIRE_PACKAGE(KinectDataTypes)

###############################################################################

MIRA_ADD_DOCUMENTATION(Evaluation
	DIRS
		doc
		include
	DEPENDS
		#RobotDataTypes
)

###############################################################################

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_LIST_DIR}/include)

###############################################################################

MIRA_ADD_LIBRARY(PUTK2Eval
	SHARED
	PACKAGE Evaluation
	SOURCE
		etc/PUTK2Eval.xml
		src/PUTK2Eval.C

	LINK_LIBS
		MIRABase
		MIRAFramework
		KinectDataTypes
		opencv_highgui
)
