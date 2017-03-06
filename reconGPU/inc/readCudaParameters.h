#ifndef _READCUDAPARAMETERS_H
#define	_READCUDAPARAMETERS_H

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <ParametersFile.h>
#include <Images.h>
#include <Geometry.h>
#include <string>
#include <vector_types.h>

// DLL export/import declaration: visibility of objects
#ifndef LINK_STATIC
  #ifdef WIN32               // Win32 build
    #ifdef DLL_BUILD    // this applies to DLL building
      #define DLLEXPORT __declspec(dllexport)
    #else                   // this applies to DLL clients/users
      #define DLLEXPORT __declspec(dllimport)
    #endif
    #define DLLLOCAL        // not explicitly export-marked objects are local by default on Win32
  #else
    #ifdef HAVE_GCCVISIBILITYPATCH   // GCC 4.x and patched GCC 3.4 under Linux
      #define DLLEXPORT __attribute__ ((visibility("default")))
      #define DLLLOCAL __attribute__ ((visibility("hidden")))
    #else
      #define DLLEXPORT
      #define DLLLOCAL
    #endif
  #endif
#else                         // static linking
  #define DLLEXPORT
  #define DLLLOCAL
#endif
#define DLLEXPORT
/* Encabezados de Funciones relacioandas con la carga de parámetros del Mlem */
DLLEXPORT int getProjectorBlockSize (string mlemFilename, string cmd, dim3* projectorBlockSize);
DLLEXPORT int getBackprojectorBlockSize (string mlemFilename, string cmd, dim3* backprojectorBlockSize);
DLLEXPORT int getPixelUpdateBlockSize(string mlemFilename, string cmd, dim3* pixelUpdateBlockSize);
DLLEXPORT int getGpuId(string mlemFilename, string cmd, int* gpuId);
DLLEXPORT bool ProcessBlockSizeString(char* strBlockSize, dim3* blockSize);
#endif
