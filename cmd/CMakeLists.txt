#  Main File for the cmd commands executables files for PET
#
#  Martin Belzunce, UTN-FRBA, Proyecto AR-PET (CNEA)
#  Copyright (c) 2010

PROJECT(cmd)

CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

###################### SOURCES CONFIGURATION ########################## 
# Códigos fuente, directorios de Include.
SET(cmd_Sources)
SET(cmd_Headers)
SET(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
SET(cmd_Sources_Dir ${cmd_SOURCE_DIR}/src)
#SET(cmd_Headers_Dir ${cmd_SOURCE_DIR}/inc)	Lo generé en el raíz.
INCLUDE(${cmd_SOURCE_DIR}/cmd.cmake OPTIONAL)
#######################################################################

###################### LINKER CONFIGURATION ###########################	
# Configuración de las dll a linkear y los directorios de búsqueda.
SET(LinkLibs recon utils data)
INCLUDE(${cmd_SOURCE_DIR}/link.cmake OPTIONAL)
#######################################################################

###################### OUTPUT CONFIGURATION ###########################
# Tipo de binario a generar (Shared Library, Static Labriry, Executable, etc), 
# y directorios de salida.
# Para el caso que quiera generar directorios distintos, para las biliotecas, los ejecutables, etc:
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${cmd_BINARY_DIR}) #Para cmake 2.6
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${cmd_BINARY_DIR})
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${cmd_BINARY_DIR})
#SET(LIBRARY_OUTPUT_PATH ${APIRL_BINARY_DIR})
message(STATUS "Output Directory: ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}")
#######################################################################


###################### BUILD CONFIGURATION ############################
#ADD_DEFINITIONS(DLL_BUILD)
# Genero un ejecutable por cada comando.
# MLEM:
ADD_Executable(MLEM ${MLEM_Sources})
# OSEM:
ADD_Executable(OSEM ${OSEM_Sources})
# GenerateImage:
ADD_Executable(generateImage ${GenerateImage_Sources})
# GenerateSystemMatrix:
ADD_Executable(generateSystemMatrix ${GenerateSystemMatrix_Sources})
# Backproject:
ADD_Executable(backproject ${Backproject_Sources})
# Project:
ADD_Executable(project ${Project_Sources})
# GenerateACFs:
ADD_Executable(generateACFs ${GenerateACFs_Source})
# GenerateSystemMatrix:
#ADD_Executable(generateSystemMatrix ${GenerateSystemMatrix_Sources})
#ADD_Executable(OSEM ${OSEM_Sources})
# Para windows se debe definir WIN32 y DLL_BUILD tambien
if(CMAKE_HOST_WIN32)
#	set_target_properties(APIRL PROPERTIES COMPILE_DEFINITIONS WIN32)
#	set_target_properties(APIRL PROPERTIES COMPILE_DEFINITIONS DLL_BUILD)
	add_definitions(/D WIN32)
endif()	
# Linkeo de DLL necesarias.
TARGET_LINK_LIBRARIES(MLEM ${LinkLibs})
TARGET_LINK_LIBRARIES(OSEM ${LinkLibs})
TARGET_LINK_LIBRARIES(generateImage ${LinkLibs})
TARGET_LINK_LIBRARIES(generateSystemMatrix ${LinkLibs})
TARGET_LINK_LIBRARIES(backproject ${LinkLibs})
TARGET_LINK_LIBRARIES(project ${LinkLibs})
TARGET_LINK_LIBRARIES(generateACFs ${LinkLibs})
#TARGET_LINK_LIBRARIES(OSEM ${LinkLibs})
# Instalación de ejecutables: install(TARGETS "todos los ejecutables" RUNTIME DESTINATION "path")
install(TARGETS MLEM RUNTIME DESTINATION "${APIRL_INSTALL_PATH}/bin")
install(TARGETS OSEM RUNTIME DESTINATION "${APIRL_INSTALL_PATH}/bin")
install(TARGETS generateImage RUNTIME DESTINATION "${APIRL_INSTALL_PATH}/bin")
install(TARGETS generateSystemMatrix RUNTIME DESTINATION "${APIRL_INSTALL_PATH}/bin")
install(TARGETS backproject RUNTIME DESTINATION "${APIRL_INSTALL_PATH}/bin")
install(TARGETS project RUNTIME DESTINATION "${APIRL_INSTALL_PATH}/bin")
install(TARGETS generateACFs RUNTIME DESTINATION "${APIRL_INSTALL_PATH}/bin")
#SET(CMAKE_CXX_FLAGS "-g3 -O0")
#SET(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0")
#SET(CMAKE_C_FLAGS "-g3 -O0")
#SET(CMAKE_C_FLAGS_DEBUG "-g3 -O0")
###########################################################################


