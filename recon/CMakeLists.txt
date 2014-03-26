#  CMAKE Main File for the recon shared library of AR-PET Image Reconstruction library (APIRL)
#
#  Martin Belzunce, UTN-FRBA, Proyecto AR-PET (CNEA)
#  Copyright (c) 2010

PROJECT(recon)
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

message(STATUS "recon_SOURCE_DIR: ${recon_SOURCE_DIR}")
message(STATUS "recon_BINARY_DIR: ${recon_BINARY_DIR}")

###################### SOURCES CONFIGURATION ########################## 
# Códigos fuente, directorios de Include.
SET(recon_Sources)
SET(recon_Headers)
SET(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
SET(recon_Sources_Dir ${recon_SOURCE_DIR}/src)
#SET(recon_Headers_Dir ${recon_SOURCE_DIR}/inc)	Lo generé en el raíz.
INCLUDE(${recon_SOURCE_DIR}/recon.cmake OPTIONAL)
#######################################################################

###################### LINKER CONFIGURATION ###########################	
# Configuración de las dll a linkear y los directorios de búsqueda.
SET(LinkLibs)
INCLUDE(${recon_SOURCE_DIR}/link.cmake OPTIONAL)
#######################################################################

###################### OUTPUT CONFIGURATION ###########################
# Tipo de binario a generar (Shared Library, Static Labriry, Executable, etc), 
# y directorios de salida.
# Para el caso que quiera generar directorios distintos, para las biliotecas, los ejecutables, etc:
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${recon_BINARY_DIR}) #Para cmake 2.6
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${recon_BINARY_DIR})
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${recon_BINARY_DIR})
#SET(LIBRARY_OUTPUT_PATH ${APIRL_BINARY_DIR})
message(STATUS "Output Directory: ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}")
#######################################################################


###################### BUILD CONFIGURATION ############################
#ADD_DEFINITIONS(DLL_BUILD)
# Genero una DLL.
ADD_Library(recon SHARED ${recon_Sources} ${recon_Headers})
# Para windows se debe definir WIN32 y DLL_BUILD tambien
if(CMAKE_HOST_WIN32)
#	set_target_properties(APIRL PROPERTIES COMPILE_DEFINITIONS WIN32)
#	set_target_properties(APIRL PROPERTIES COMPILE_DEFINITIONS DLL_BUILD)
	add_definitions(/D WIN32)
	add_definitions(/D DLL_BUILD)
endif()	
# Linkeo de DLL necesarias.
TARGET_LINK_LIBRARIES(recon ${LinkLibs})

#INSTALL(TARGETS APIRL LIBRARY DESTINATION  build)
if(CMAKE_HOST_WIN32)
	install(TARGETS recon RUNTIME DESTINATION "${APIRL_INSTALL_PATH}/bin")
else()
	install(TARGETS recon LIBRARY DESTINATION "${APIRL_INSTALL_PATH}/bin")
endif()

#SET(CMAKE_CXX_FLAGS "-g3 -O0")
#SET(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0")
#SET(CMAKE_C_FLAGS "-g3 -O0")
#SET(CMAKE_C_FLAGS_DEBUG "-g3 -O0")
###########################################################################

