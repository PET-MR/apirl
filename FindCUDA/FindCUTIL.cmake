#  FindCUTIL: complemento para FindCUDA cmake files.
#  Realiza la búsqueda de los directorios de cutil.h y cutil32.lib.
#  Debe llamarse previamente a FindCUDA.
#  El script genera las siguientes variables:
#	CUDA_CUT_INCLUDE_DIR: Directorio donde se encuentran los headers de la librería cutil.
#	CUDA_CUT_LIBRARY: Path completo de la librería.
#  Martin Belzunce, UTN-FRBA, Proyecto AR-PET (CNEA)
#  Copyright (c) 2010

# Primero el .h.
# message(status $ENV{CUDA_SDK_INSTALL_PATH})
# Agrego paths para poder encontrar la SDK en windows 7.
find_path(CUDA_CUT_INCLUDE_DIR
  cutil.h
  PATHS $ENV{CUDA_SDK_INSTALL_PATH} 
		${CUDA_SDK_SEARCH_PATH} 
		"C:/CUDA/NVIDIA GPU Computing SDK"
		"C:/ProgramData/NVIDIA Corporation/NVIDIA GPU Computing SDK 3.2/C/common/inc"
  PATH_SUFFIXES "common/inc" "C/common/inc"
  DOC "Location of cutil.h"
  NO_DEFAULT_PATH
  )
# Now search system paths
find_path(CUDA_CUT_INCLUDE_DIR cutil.h DOC "Location of cutil.h")

mark_as_advanced(CUDA_CUT_INCLUDE_DIR)


# Ahora el .lib.
# cutil library is called cutil64 for 64 bit builds on windows.  We don't want
# to get these confused, so we are setting the name based on the word size of
# the build.
# Originalmente el cuda_cutil_name lo seteabamos fijo a cutil64  o cutil32
# según el resultado del siguiente test. Pero para el vaso en que el visual
# studio es de 32 bits y se ejecuta sobre windows 7 (64 bits), el resultado
# del test da cutil32 en vez de cutil64. Por lo que seteamos los dos nombres.
if(CMAKE_HOST_WIN32 OR CMAKE_HOST_WIN64)
	if(CMAKE_SIZEOF_VOID_P EQUAL 8)
	  set(cuda_cutil_name cutil64 cutil32)
	else(CMAKE_SIZEOF_VOID_P EQUAL 4)
	  set(cuda_cutil_name cutil32 cutil64)
	endif(CMAKE_SIZEOF_VOID_P EQUAL 8)
else()
	if(CMAKE_SIZEOF_VOID_P EQUAL 8)
	  set(cuda_cutil_name cutil_x86_64)
	else(CMAKE_SIZEOF_VOID_P EQUAL 4)
	  set(cuda_cutil_name cutil_x86_32)
	endif(CMAKE_SIZEOF_VOID_P EQUAL 8)
endif()
message(${cuda_cutil_name})
message($ENV{CUDA_SDK_INSTALL_PATH} )

find_library(CUDA_CUT_LIBRARY
  NAMES cutil ${cuda_cutil_name}
  PATHS $ENV{CUDA_SDK_INSTALL_PATH} 
		${CUDA_SDK_SEARCH_PATH} 
		"C:/CUDA/NVIDIA GPU Computing SDK"
		"C:/ProgramData/NVIDIA Corporation/NVIDIA GPU Computing SDK 3.2"
  # The new version of the sdk shows up in common/lib, but the old one is in lib
  PATH_SUFFIXES "common/lib" "lib" "C/common/lib" "C/lib"
  DOC "Location of cutil library"
  NO_DEFAULT_PATH
  )
# Now search system paths
find_library(CUDA_CUT_LIBRARY NAMES cutil ${cuda_cutil_name} DOC "Location of cutil library")
mark_as_advanced(CUDA_CUT_LIBRARY)
set(CUDA_CUT_LIBRARIES ${CUDA_CUT_LIBRARY})
