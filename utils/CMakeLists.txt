#  CMAKE Main File for the utils shared library of AR-PET Image Reconstruction library (APIRL)
#
#  Martin Belzunce, UTN-FRBA, Proyecto AR-PET (CNEA)
#  Copyright (c) 2010

PROJECT(utils)
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

message(STATUS "utils_SOURCE_DIR: ${utils_SOURCE_DIR}")
message(STATUS "utils_BINARY_DIR: ${utils_BINARY_DIR}")

###################### SOURCES CONFIGURATION ########################## 
# Códigos fuente, directorios de Include.
SET(utils_Sources)
SET(utils_Headers)
SET(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
SET(utils_Sources_Dir ${utils_SOURCE_DIR}/src)
#SET(utils_Headers_Dir ${utils_SOURCE_DIR}/inc) Lo generé en el raíz.
INCLUDE(${utils_SOURCE_DIR}/utils.cmake OPTIONAL)
#######################################################################

###################### LINKER CONFIGURATION ###########################	
# Configuración de las dll a linkear y los directorios de búsqueda.
SET(LinkLibs)
INCLUDE(${utils_SOURCE_DIR}/link.cmake OPTIONAL)
#######################################################################

###################### OUTPUT CONFIGURATION ###########################
# Tipo de binario a generar (Shared Library, Static Labriry, Executable, etc), 
# y directorios de salida.
# Para el caso que quiera generar directorios distintos, para las biliotecas, los ejecutables, etc:
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${utils_BINARY_DIR}) #Para cmake 2.6
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${utils_BINARY_DIR})
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${utils_BINARY_DIR})
#SET(LIBRARY_OUTPUT_PATH ${APIRL_BINARY_DIR})
message(STATUS "Output Directory: ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}")
#######################################################################


###################### BUILD CONFIGURATION ############################
#ADD_DEFINITIONS(DLL_BUILD)
# Genero una DLL.
ADD_Library(utils SHARED ${utils_Sources} ${utils_Headers})
# Para windows se debe definir WIN32 y DLL_BUILD tambien
if(CMAKE_HOST_WIN32)
#	set_target_properties(APIRL PROPERTIES COMPILE_DEFINITIONS WIN32)
#	set_target_properties(APIRL PROPERTIES COMPILE_DEFINITIONS DLL_BUILD)
	add_definitions(/D WIN32)
	add_definitions(/D DLL_BUILD)
#	SET_SOURCE_FILES_PROPERTIES(${utils_Sources} ${utils_Headers} PROPERTIES COMPILE_DEFINITIONS "DLL_BUILD")
endif()	
# Linkeo de DLL necesarias.
TARGET_LINK_LIBRARIES(utils ${LinkLibs})

#INSTALL(TARGETS APIRL LIBRARY DESTINATION  build)
if(CMAKE_HOST_WIN32)
	install(TARGETS utils RUNTIME DESTINATION "${APIRL_INSTALL_PATH}/bin")
else()
	install(TARGETS utils LIBRARY DESTINATION "${APIRL_INSTALL_PATH}/bin")
endif()
#SET(CMAKE_CXX_FLAGS "-g3 -O0")
#SET(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0")
#SET(CMAKE_C_FLAGS "-g3 -O0")
#SET(CMAKE_C_FLAGS_DEBUG "-g3 -O0")
###########################################################################

