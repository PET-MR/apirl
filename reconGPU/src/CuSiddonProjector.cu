/**
	\file CuSiddonProjector.cpp
	\brief Archivo que contiene la implementación de la clase CuSiddonProjector.
	Es el proyector de siddon implementado en Cuda.
	\todo 
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2014.07.11
	\version 1.1.0
*/

#include <CuSiddonProjector.h>

CuSiddonProjector::CuSiddonProjector()
{
  this->numSamplesOnDetector = 1;  
}

CuSiddonProjector::CuSiddonProjector(unsigned int numThreadsPerBlockX, unsigned int numThreadsPerBlockY, unsigned int numThreadsPerBlockZ, 
			     unsigned int numBlocksX, unsigned int numBlocksY, unsigned int numBlocksZ)
{
  /* Configuro el kernel de ejecución. */
  this->blockSize = dim3(numThreadsPerBlockX, numThreadsPerBlockY, numThreadsPerBlockZ);
  this->gridSize = dim3(numBlocksX, numBlocksY, numBlocksZ);
}


bool CuSiddonProjector::Backproject (Sinogram2D* InputSinogram, Image* outputImage)
{
  
  return true;
}

/// Sobrecarga que realiza la Backprojection de InputSinogram/EstimatedSinogram
bool CuSiddonProjector::DivideAndBackproject (Sinogram2D* InputSinogram, Sinogram2D* EstimatedSinogram, Image* outputImage)
{
  
  return true;
}

bool CuSiddonProjector::Project (Image* inputImage, Sinogram2D* outputProjection)
{
  
  return true;
}


bool CuSiddonProjector::Backproject (Sinogram2Dtgs* InputSinogram, Image* outputImage)
{
  return true;
}

/// Sobrecarga que realiza la Backprojection de InputSinogram/EstimatedSinogram
bool CuSiddonProjector::DivideAndBackproject (Sinogram2Dtgs* InputSinogram, Sinogram2Dtgs* EstimatedSinogram, Image* outputImage)
{

  return true;
}

bool CuSiddonProjector::Project (Image* inputImage, Sinogram2Dtgs* outputProjection)
{

  return true;
}

/** Sección para Sinogram3D. */
bool CuSiddonProjector::Project (Image* inputImage, Sinogram3D* outputProjection)
{
  /* Este método simplemente encapsula la llamada al kernel.
    El tamaño de la ejecución del kernel está definida en las propiedades gridSize y blockSize de la clase.
    La misma se configura en el constructor o con el método setKernelConfig.
    */
  CUDA_Forward_Projection<<<gridSize, blockSize>>>(cuda_volume, cuda_projected_michelogram, cuda_michelogram);
  /// Sincronización de todos los threads.
  cudaThreadSynchronize();
  return true;
}


bool CuSiddonProjector::Backproject (Sinogram3D* inputProjection, Image* outputImage)
{
  return true;
}

/// Sobrecarga que realiza la Backprojection del cociente InputSinogram3D/EstimatedSinogram3D
bool CuSiddonProjector::DivideAndBackproject (Sinogram3D* InputSinogram3D, Sinogram3D* EstimatedSinogram3D, Image* outputImage)
{
  
  return true;
}