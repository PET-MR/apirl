#  CMAKE Main File for the data shared library of AR-PET Image Reconstruction library (APIRL)
#
#  Martin Belzunce, UTN-FRBA, Proyecto AR-PET (CNEA)
#  Copyright (c) 2010

PROJECT(data)
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

# Por default compilo en modo Release.
IF(NOT CMAKE_BUILD_TYPE )
   SET( CMAKE_BUILD_TYPE "Release" )
ENDIF()

message(STATUS "data_SOURCE_DIR: ${data_SOURCE_DIR}")
message(STATUS "data_BINARY_DIR: ${data_BINARY_DIR}")

###################### SOURCES CONFIGURATION ########################## 
# Códigos fuente, directorios de Include.
SET(data_Sources)
SET(data_Headers)
SET(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
SET(data_Sources_Dir ${data_SOURCE_DIR}/src)
SET(Medcon_Dir ${data_SOURCE_DIR}/medcon)
# SET(data_Headers_Dir ${data_SOURCE_DIR}/inc) Lo generé en el raíz.
INCLUDE(${data_SOURCE_DIR}/data.cmake OPTIONAL)
#######################################################################

###################### LINKER CONFIGURATION ###########################	
# Configuración de las dll a linkear y los directorios de búsqueda.
SET(LinkLibs)
INCLUDE(${data_SOURCE_DIR}/link.cmake OPTIONAL)
#######################################################################

###################### OUTPUT CONFIGURATION ###########################
# Tipo de binario a generar (Shared Library, Static Labriry, Executable, etc), 
# y directorios de salida.
# Para el caso que quiera generar directorios distintos, para las biliotecas, los ejecutables, etc:
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${data_BINARY_DIR}) #Para cmake 2.6
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${data_BINARY_DIR})
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${data_BINARY_DIR})
#SET(LIBRARY_OUTPUT_PATH ${APIRL_BINARY_DIR})
message(STATUS "Output Directory: ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}")
#######################################################################


###################### BUILD CONFIGURATION ############################
#ADD_DEFINITIONS(DLL_BUILD)
SET(BUILD_SHARED_LIBS ON)
# Genero una DLL.
ADD_Library(data SHARED ${data_Sources} ${data_Headers} ${Medcon_Sources})
# Para windows se debe definir WIN32 y DLL_BUILD tambien
if(CMAKE_HOST_WIN32)
#	set_target_properties(APIRL PROPERTIES COMPILE_DEFINITIONS WIN32)
#	set_target_properties(APIRL PROPERTIES COMPILE_DEFINITIONS DLL_BUILD)
	add_definitions(/D WIN32)
	add_definitions(/D DLL_BUILD)
endif()	
# Linkeo de DLL necesarias.
TARGET_LINK_LIBRARIES(data ${LinkLibs})

#INSTALL(TARGETS APIRL LIBRARY DESTINATION  build)
if(CMAKE_HOST_WIN32)
	install(TARGETS data RUNTIME DESTINATION "${APIRL_INSTALL_PATH}/bin")
else()
	install(TARGETS data LIBRARY DESTINATION "${APIRL_INSTALL_PATH}/bin")
endif()
# Definiciones de tamaños de variables necesarias para módulo de Medcon. También se podría usar para otras configuraciones en el futuro.
# En Windows es necesario el /D para agregar un define de pre procesador (?, verificar).
if(CMAKE_HOST_WIN32)
  if(CMAKE_SIZEOF_VOID_P EQUAL 4)
	  add_definitions(/D X86)
  elseif(CMAKE_SIZEOF_VOID_P EQUAL 8)
	  add_definitions(/D AMD64)
  endif()
else()
  if(CMAKE_SIZEOF_VOID_P EQUAL 4)
	  add_definitions(-DX86)
  elseif(CMAKE_SIZEOF_VOID_P EQUAL 8)
	  add_definitions(-DAMD64)
  endif()
endif()
# Opciones de compilación. Para el debug incluimos información de debuggeo y baja optimización,
# mientras que para el release compilamos con toda la optimización.
#SET(CMAKE_CXX_FLAGS "-g3 -O0")
#SET(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0")
#SET(CMAKE_CXX_FLAGS_RELEASE "-O2 -pipe -march=core2" )
#SET(CMAKE_C_FLAGS "-g3 -O0")
#SET(CMAKE_C_FLAGS_DEBUG "-g3 -O0")
###########################################################################

