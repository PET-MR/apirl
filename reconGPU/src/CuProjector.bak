/**
	\file CuProjector.cpp
	\brief Archivo que contiene la implementación de la clase abstracta CuProjector.
	Es un proyector genérico implementado en Cuda.
	\todo 
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2014.07.11
	\version 1.1.0
*/

#include <CuProjector.h>

CuProjector::CuProjector()
{
}

CuProjector::CuProjector(unsigned int numThreadsPerBlockX, unsigned int numThreadsPerBlockY, unsigned int numThreadsPerBlockZ, 
			     unsigned int numBlocksX, unsigned int numBlocksY, unsigned int numBlocksZ)
{
  /* Configuro el kernel de ejecución. */
  this->blockSize = dim3(numThreadsPerBlockX, numThreadsPerBlockY, numThreadsPerBlockZ);
  this->gridSize = dim3(numBlocksX, numBlocksY, numBlocksZ);
}

