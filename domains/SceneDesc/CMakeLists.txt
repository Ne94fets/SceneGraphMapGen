
MIRA_REQUIRE_PACKAGE(SceneDesc)

###############################################################################

MIRA_ADD_DOCUMENTATION(SceneDesc
	DIRS
		doc
		include
	DEPENDS
        RecognitionDataTypes
)

###############################################################################

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_LIST_DIR}/include)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/toolboxes/RecognitionDataTypes/include)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/toolboxes/KinectDataTypes/include)

###############################################################################

MIRA_ADD_LIBRARY(GraphMap
    SHARED
    PACKAGE SceneDesc
    SOURCE
        etc/GraphMap.xml
        src/GraphMap.C

    LINK_LIBS
        MIRABase
        MIRAFramework
        KinectDataTypes
        RecognitionDataTypes
        neo4j-client
)
