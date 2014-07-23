#  Main File for the cmdGPU executables files for PET
#
#  Martin Belzunce, UTN-FRBA, Proyecto AR-PET (CNEA)
#  Copyright (c) 2010

PROJECT(cmdGPU)

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
# Códigos fuente, directorios de Include.
SET(cmdGPU_Sources)
SET(cmdGPU_Headers)
SET(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
SET(cmdGPU_Sources_Dir ${cmdGPU_SOURCE_DIR}/src)
SET(cmdGPU_Headers_Dir ${cmdGPU_SOURCE_DIR}/inc)
INCLUDE(${cmdGPU_SOURCE_DIR}/cmdGPU.cmake OPTIONAL)
#######################################################################

###################### LINKER CONFIGURATION ###########################	
# Configuración de las dll a linkear y los directorios de búsqueda.
SET(LinkLibs)
INCLUDE(${cmdGPU_SOURCE_DIR}/link.cmake OPTIONAL)
#######################################################################

###################### OUTPUT CONFIGURATION ###########################
# Tipo de binario a generar (Shared Library, Static Labriry, Executable, etc), 
# y directorios de salida.
# Para el caso que quiera generar directorios distintos, para las biliotecas, los ejecutables, etc:
SET(CUDA_GENERATED_OUTPUT_DIR ${cmdGPU_BINARY_DIR})
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${cmdGPU_BINARY_DIR}) #Para cmake 2.6
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${cmdGPU_BINARY_DIR})
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${cmdGPU_BINARY_DIR})
message(STATUS "Output Directory: ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}")
#######################################################################


###################### BUILD CONFIGURATION ############################
# Genero un ejecutable por cada comando.
CUDA_ADD_EXECUTABLE(cuMLEM ${cuMLEM_Sources})
# Para windows se debe definir WIN32 y DLL_BUILD tambien
if(CMAKE_HOST_WIN32)
#	set_target_properties(APIRL PROPERTIES COMPILE_DEFINITIONS WIN32)
# Add definitions to the nvcc definition line.
	add_definitions(/D WIN32)
	SET(CUDA_64_BIT_DEVICE_CODE OFF)
	if (CUDA_LIBRARIES_BUILD STREQUAL STATIC)
		add_definitions(/D LINK_STATIC)
	endif()
	if (CMAKE_BUILD_TYPE STREQUAL DEBUG)
		add_definitions(/D _DEBUG)
	endif()
endif()	
# Linkeo de DLL necesarias.
TARGET_LINK_LIBRARIES(cuMLEM ${LinkLibs})

# Instalación de ejecutables: install(TARGETS "todos los ejecutables" RUNTIME DESTINATION "path")
install(TARGETS cuMLEM RUNTIME DESTINATION "${APIRL_INSTALL_PATH}/bin")
#SET(CMAKE_CXX_FLAGS "-g3 -O0")
#SET(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0")
#SET(CMAKE_C_FLAGS "-g3 -O0")
#SET(CMAKE_C_FLAGS_DEBUG "-g3 -O0")
###########################################################################


