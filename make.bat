@echo OFF

cmake -P FindMIRARoot.cmake > root_dir.tmp
SET /P MIRA_ROOT_DIR=<root_dir.tmp
del root_dir.tmp

nmake /f Makefile.nmake /nologo MIRA_ROOT_DIR=%MIRA_ROOT_DIR% %1 %2 %3 %4 %5 %6 %7 %8 %9
