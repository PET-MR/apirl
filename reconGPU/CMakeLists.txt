#  CMAKE Main File for the reconGPU shared library of AR-PET Image Reconstruction library (APIRL)
#
#  Martin Belzunce, UTN-FRBA, Proyecto AR-PET (CNEA)
#  Copyright (c) 2010

PROJECT(reconGPU)
SET(BUILD_SHARED_LIBS  ON)
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

###################### CUDA CONFIGURATION #############################
# Ejecuto el FindCUDA.
#SET(FIND_CUDA_DIR ${APIRL_SOLUTION_SOURCE_DIR}/FindCUDA)
#INCLUDE(${FIND_CUDA_DIR}/CMake/cuda/FindCUDA.cmake)
set(CMAKE_MODULE_PATH "${FindCUDA_Dir}/CMake/cuda" ${CMAKE_MODULE_PATH})
find_package(CUDA REQUIRED FIND_CUDA_DIR)
# Llamo al script FindCUTIL que me genera las variables CUDA_CUT_INCLUDE_DIR y CUDA_CUT_LIBRARIES.
#INCLUDE(${FindCUDA_Dir}/FindCUTIL.cmake)
#message(STATUS "CUDA_CUT_INCLUDE_DIR: ${CUDA_CUT_INCLUDE_DIR}")
ADD_DEFINITIONS(-D_CRT_SECURE_NO_WARNINGS)
SET(CUDA_BUILD_CUBIN ON)
SET(CUDA_VERBOSE_BUILD ON)
#######################################################################

###################### SOURCES CONFIGURATION ########################## 
# C√≥digos fuente, directorios de Include.
SET(reconGPU_Sources)
SET(reconGPU_Headers)
SET(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
SET(reconGPU_Sources_Dir ${reconGPU_SOURCE_DIR}/src)
SET(reconGPU_Headers_Dir ${reconGPU_SOURCE_DIR}/inc)
message(STATUS "reconGPU_Headers_Dir: ${reconGPU_Headers_Dir}")
INCLUDE(${reconGPU_SOURCE_DIR}/reconGPU.cmake OPTIONAL)
#######################################################################

###################### LINKER CONFIGURATION ###########################	
# Configuraci√≥n de las dll a linkear y los directorios de b√∫squeda.
SET(LinkLibs)
INCLUDE(${reconGPU_SOURCE_DIR}/link.cmake OPTIONAL)
#######################################################################

###################### OUTPUT CONFIGURATION ###########################
# Tipo de binario a generar (Shared Library, Static Labriry, Executable, etc), 
# y directorios de salida.
# Para el caso que quiera generar directorios distintos, para las biliotecas, los ejecutables, etc:
SET(CUDA_GENERATED_OUTPUT_DIR ${reconGPU_BINARY_DIR})
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${reconGPU_BINARY_DIR}) #Para cmake 2.6
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${reconGPU_BINARY_DIR})
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${reconGPU_BINARY_DIR})
message(STATUS "Output Directory: ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}")
#######################################################################


###################### BUILD CONFIGURATION ############################
#ADD_DEFINITIONS(DLL_BUILD)
# Genero una DLL.
#ADD_Library(reconGPU SHARED ${reconGPU_Sources} ${reconGPU_Headers})${CUDA_LIBRARIES_BUILD}
#CUDA_ADD_LIBRARY(reconGPU  ${CUDA_LIBRARIES_BUILD} ${reconGPU_Sources} ${reconGPU_cuSources} ${reconGPU_Headers} ${reconGPU_cuHeaders})
# Genero varias DLLs, una por cada proyector y los m®¶todos de reconstrucci®Æn:
CUDA_ADD_LIBRARY(reconGPU  "src/CuSiddonProjector.cu" "src/CuProjector.cpp")
CUDA_ADD_LIBRARY(cuOsem  "src/CuOsemSinogram3d.cu")
CUDA_ADD_LIBRARY(cuMlem  "src/CuMlemSinogram3d.cu")
#CUDA_ADD_LIBRARY(reconGPU2  "src/CuMlemSinogram3d.cu")
# Para windows se debe definir WIN32 y DLL_BUILD tambien
if(CMAKE_HOST_WIN32)
#	set_target_properties(APIRL PROPERTIES COMPILE_DEFINITIONS WIN32)
#	set_target_properties(APIRL PROPERTIES COMPILE_DEFINITIONS DLL_BUILD)
# Add definitions to the nvcc definition line.
	add_definitions(/D WIN32)
	add_definitions(/D DLL_BUILD)
	SET(CUDA_64_BIT_DEVICE_CODE OFF)
	if (CUDA_LIBRARIES_BUILD STREQUAL STATIC)
		add_definitions(/D LINK_STATIC)
	endif()
	if (CMAKE_BUILD_TYPE STREQUAL DEBUG)
		add_definitions(/D _DEBUG)
	endif()
endif()	
# Si compilo en Fermi defino la variable FERMI. Esto me permite generar c®Ædigo
# condicional que solo anda en esa arquitectura. Por ejemplo, para utilizar printf
# dentro de los kernels:
if(CUDA_COMPUTE_CAPABILITY STREQUAL 2.0)
  add_definitions(-D_FERMI)
endif()
# Linkeo de DLL necesarias.
TARGET_LINK_LIBRARIES(reconGPU ${LinkLibs})


if(CMAKE_HOST_WIN32)
	if(CUDA_LIBRARIES_BUILD STREQUAL STATIC)
		install(TARGETS reconGPU cuOsem cuMlem ARCHIVE DESTINATION "${APIRL_INSTALL_PATH}/bin")
	else()
		install(TARGETS reconGPU cuOsem cuMlem RUNTIME DESTINATION "${APIRL_INSTALL_PATH}/bin")
	endif()
else()
	if(CUDA_LIBRARIES_BUILD STREQUAL STATIC)
		install(TARGETS reconGPU cuOsem cuMlem ARCHIVE DESTINATION "${APIRL_INSTALL_PATH}/bin")
	else()
		install(TARGETS reconGPU cuOsem cuMlem LIBRARY DESTINATION "${APIRL_INSTALL_PATH}/bin")
	endif()
endif()

# Por default compilamos en modo device, para usar modo emulaci√≥n, cambiar esta variable.
SET(CUDA_BUILD_EMULATION OFF)
#SET(CMAKE_CXX_FLAGS "-g3 -O0")
#SET(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0")
#SET(CMAKE_C_FLAGS "-g3 -O0")
#SET(CMAKE_C_FLAGS_DEBUG "-g3 -O0")
###########################################################################


