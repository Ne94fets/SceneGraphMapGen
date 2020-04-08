
MIRA_REQUIRE_PACKAGE(Recognition)

###############################################################################

MIRA_ADD_DOCUMENTATION(Recognition
	DIRS
		doc
		include
	DEPENDS
		#RobotDataTypes
)

###############################################################################

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_LIST_DIR}/include)

###############################################################################

MIRA_ADD_LIBRARY(ObjectRecognition3d
	SHARED
	PACKAGE Recognition
	SOURCE
		etc/ObjectRecognition3d.xml
		src/ObjectRecognition3d.C

	LINK_LIBS
		MIRABase
		MIRAFramework
		libtensorflow_cc.so
)
