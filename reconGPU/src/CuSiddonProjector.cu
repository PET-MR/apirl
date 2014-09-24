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
#include "../kernels/CuSiddonProjector_kernels.cu"

CuSiddonProjector::CuSiddonProjector()
{
  this->numSamplesOnDetector = 1;  
}


bool CuSiddonProjector::InitGpuMemory(Sinogram3DCylindricalPet* inputSinogram)
{
//   checkCudaErrors(cudaMemcpyToSymbol(cuda_threads_per_block, &(blockSizeProjector.x), sizeof(unsigned int)));
//   checkCudaErrors(cudaMemcpyToSymbol(cuda_threads_per_block_update_pixel, &(blockSizeImageUpdate.x), sizeof(unsigned int)));
//   checkCudaErrors(cudaMemcpyToSymbol(cuda_nr_splitter, &NR_Splitter, sizeof(unsigned int)));
//   checkCudaErrors(cudaMemcpyToSymbol(cuda_rows_splitter, &rowSplitter, sizeof(unsigned int)));
  float aux = inputSinogram->getRadioScanner_mm();
  checkCudaErrors(cudaMemcpyToSymbol(d_RadioScanner_mm, &aux, sizeof(inputSinogram->getRadioScanner_mm())));
}

/** Sección para Sinogram3D. */
bool CuSiddonProjector::Project (float* d_image, float* d_projection, float *d_ring1, float *d_ring2, Image* inputImage, Sinogram3DCylindricalPet* outputSinogram, bool copyResult)
{
  /* Este método simplemente encapsula la llamada al kernel.
    El tamaño de la ejecución del kernel está definida en las propiedades gridSize y blockSize de la clase.
    La misma se configura en el constructor o con el método setKernelConfig.
    */
  cuSiddonProjection<<<gridSize, blockSize>>>(d_image, d_projection, d_ring1, d_ring2, outputSinogram->getNumR(), outputSinogram->getNumProj(), outputSinogram->getNumRings(), outputSinogram->getNumSinograms());
  /// Sincronización de todos los threads.
  checkCudaErrors(cudaThreadSynchronize());
  return true;
}

bool CuSiddonProjector::DivideAndBackproject (float* d_inputSinogram, float* d_estimatedSinogram, float* d_outputImage, float *d_ring1, float *d_ring2, Sinogram3DCylindricalPet* inputSinogram, Image* outputImage, bool copyResult)
{
  /* Este método simplemente encapsula la llamada al kernel.
    El tamaño de la ejecución del kernel está definida en las propiedades gridSize y blockSize de la clase.
    La misma se configura en el constructor o con el método setKernelConfig.
    */
  cuSiddonDivideAndBackproject<<<gridSize, blockSize>>>(d_inputSinogram, d_estimatedSinogram, d_outputImage, 
					     d_ring1, d_ring2, inputSinogram->getNumR(), inputSinogram->getNumProj(), inputSinogram->getNumRings(), inputSinogram->getNumSinograms());
  /// Sincronización de todos los threads.
  checkCudaErrors(cudaThreadSynchronize());
  return true;
}

bool CuSiddonProjector::Backproject (float * d_inputSinogram, float* d_outputImage, float *d_ring1, float *d_ring2, Sinogram3DCylindricalPet* inputSinogram, Image* outputImage, bool copyResult)
{
  /* Este método simplemente encapsula la llamada al kernel.
    El tamaño de la ejecución del kernel está definida en las propiedades gridSize y blockSize de la clase.
    La misma se configura en el constructor o con el método setKernelConfig.
    */
  cuSiddonBackprojection<<<gridSize, blockSize>>>(d_inputSinogram, d_outputImage, d_ring1, d_ring2, 
							inputSinogram->getNumR(), inputSinogram->getNumProj(), inputSinogram->getNumRings(), inputSinogram->getNumSinograms());
  /// Sincronización de todos los threads.
  checkCudaErrors(cudaThreadSynchronize());
  return true;
}



	
// bool CuSiddonProjector::Backproject (Sinogram3D* inputProjection, Image* outputImage)
// {
//   return true;
// }
// 
// /// Sobrecarga que realiza la Backprojection del cociente InputSinogram3D/EstimatedSinogram3D
// bool CuSiddonProjector::DivideAndBackproject (Sinogram3D* InputSinogram3D, Sinogram3D* EstimatedSinogram3D, Image* outputImage)
// {
//   
//   return true;
// }
// 
// bool CuSiddonProjector::Backproject (Sinogram2D* InputSinogram, Image* outputImage)
// {
//   
//   return true;
// }
// 
// /// Sobrecarga que realiza la Backprojection de InputSinogram/EstimatedSinogram
// bool CuSiddonProjector::DivideAndBackproject (Sinogram2D* InputSinogram, Sinogram2D* EstimatedSinogram, Image* outputImage)
// {
//   
//   return true;
// }
// 
// bool CuSiddonProjector::Project (Image* inputImage, Sinogram2D* outputProjection)
// {
//   
//   return true;
// }
// 
// 
// bool CuSiddonProjector::Backproject (Sinogram2Dtgs* InputSinogram, Image* outputImage)
// {
//   return true;
// }
// 
// /// Sobrecarga que realiza la Backprojection de InputSinogram/EstimatedSinogram
// bool CuSiddonProjector::DivideAndBackproject (Sinogram2Dtgs* InputSinogram, Sinogram2Dtgs* EstimatedSinogram, Image* outputImage)
// {
// 
//   return true;
// }
// 
// bool CuSiddonProjector::Project (Image* inputImage, Sinogram2Dtgs* outputProjection)
// {
// 
//   return true;
// }