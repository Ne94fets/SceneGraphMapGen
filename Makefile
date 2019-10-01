TOP_DIR=$(CURDIR)

MIRA_ROOT_DIR=$(shell cmake -P FindMIRARoot.cmake)
MIRA_MAKE_DIR=$(MIRA_ROOT_DIR)/make
-include $(MIRA_ROOT_DIR)/make/Insource.make
