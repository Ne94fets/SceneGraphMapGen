# Examines each path in MIRA_PATH in order to find the MIRA root directory, which
# is identified by the file "mira.root"

set(FindMIRARoot_MIRA_PATH $ENV{MIRA_PATH})
if(UNIX)
	# note: the "" around "${MIRA_PATH}" are very important !
	string(REPLACE ":" ";" FindMIRARoot_MIRA_PATH "${FindMIRARoot_MIRA_PATH}")
endif()

set(FOUND_MIRA_ROOT_DIR 0)
set(MIRA_ROOT_DIR "")
FOREACH(path ${FindMIRARoot_MIRA_PATH})
	GET_FILENAME_COMPONENT(pathComponent ${path} ABSOLUTE)
	# strip any trailing slashes from every path in MIRA_PATH env
	#MESSAGE(STATUS "Examining ${pathComponent}")

	if(EXISTS "${pathComponent}/mira.root")
		set(FOUND_MIRA_ROOT_DIR 1)
		set(MIRA_ROOT_DIR ${pathComponent})
	endif()
ENDFOREACH(path)

IF(FOUND_MIRA_ROOT_DIR)
	# return the final relative path; must use echo, as cmake does not provide 
	# mechanism to output to stdout :( 
	execute_process(COMMAND ${CMAKE_COMMAND} -E echo ${MIRA_ROOT_DIR})
ELSE()
	message("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
	message("Your MIRA_PATH environment variable is not set properly.\nPlease add the path of the MIRA core system to your MIRA_PATH environment variable.")
	message("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!") 
	message(FATAL_ERROR)
ENDIF()
