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

/* Encabezados de Funciones relacioandas con la carga de parámetros del Mlem */
int getProjectorBlockSize (string mlemFilename, string cmd, dim3* projectorBlockSize);
int getBackprojectorBlockSize (string mlemFilename, string cmd, dim3* backprojectorBlockSize);
int getPixelUpdateBlockSize(string mlemFilename, string cmd, dim3* pixelUpdateBlockSize);
int getGpuId(string mlemFilename, string cmd, int* gpuId);
bool ProcessBlockSizeString(char* strBlockSize, dim3* blockSize);
#endif
