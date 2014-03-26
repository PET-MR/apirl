#include <CudaMlem.h>

/// Para el CudaMlem no tengo en cuenta los proyectores de la clase Mlem, por eso le paso null.
CudaMlem::CudaMlem(Image* cInitialEstimate, string cPathSalida, string cOutputPrefix, int cNumIterations, int cSaveIterationInterval, bool cSensitivityImageFromFile, dim3 blockSizeProj, dim3 blockSizeBackproj, dim3 blockSizeIm) : Mlem( cInitialEstimate, cPathSalida, cOutputPrefix, cNumIterations, cSaveIterationInterval, cSensitivityImageFromFile, NULL, NULL)
{
  blockSizeProjector = blockSizeProj;
  blockSizeBackprojector = blockSizeBackproj;
  blockSizeImageUpdate = blockSizeIm;
}

void CudaMlem::setExecutionConfigForProjectorKernel(dim3 blockSizeProj)
{
  blockSizeProjector = blockSizeProj;
}
void CudaMlem::setExecutionConfigForBackrojectorKernel(dim3 blockSizeBackproj)
{
  blockSizeBackprojector = blockSizeBackproj;
}

void CudaMlem::setExecutionConfigForImageKernel(dim3 blockSizeIm)
{
  blockSizeImageUpdate = blockSizeIm;
}

bool CudaMlem::initCuda (int device, Logger* logger)
{
  int deviceCount;
  int driverVersion;
  int runtimeVersion;
  char c_string[512];
  //Check that I can multiplie the matrix
  cudaGetDeviceCount(&deviceCount);
  if(device>=deviceCount)
	  return false;

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device);
  /// Info del driver:
  cudaDriverGetVersion(&driverVersion);
  cudaRuntimeGetVersion(&runtimeVersion);
  
  strcpy(c_string, "______GPU PROPERTIES_____");
  logger->writeLine(c_string, strlen(c_string));
  logger->writeValue(" Nombre", deviceProp.name);
  
  sprintf(c_string, "%d", driverVersion);
  logger->writeValue(" Cuda Driver Version", c_string);
  sprintf(c_string, "%d", runtimeVersion);
  logger->writeValue(" Cuda Runtime Version", c_string);
  
  sprintf(c_string, "%d", deviceProp.computeMode);
  logger->writeValue(" Compute Mode", c_string);
  
  sprintf(c_string, "%d", deviceProp.totalGlobalMem);
  logger->writeValue(" Total Global Memory", c_string);
  
  sprintf(c_string, "%d", deviceProp.sharedMemPerBlock);
  logger->writeValue(" Shared Memory per Block", c_string);

  sprintf(c_string, "%d", deviceProp.regsPerBlock);
  logger->writeValue(" Register pero Block", c_string);

  sprintf(c_string, "%d", deviceProp.warpSize);
  logger->writeValue(" Warp Size", c_string);

  sprintf(c_string, "%d", deviceProp.maxThreadsPerBlock);
  logger->writeValue(" Max Threads Per Block", c_string);

  sprintf(c_string, "%dx%dx%d", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
  logger->writeValue(" Max Threads Dimension", c_string);
  
  sprintf(c_string, "%dx%dx%d", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
  logger->writeValue(" Max Grid Size", c_string);
  
  sprintf(c_string, "%d", deviceProp.maxThreadsPerBlock);
  logger->writeValue(" Max Threads Per Block", c_string);

  sprintf(c_string, "%d", deviceProp.clockRate);
  logger->writeValue(" Clock Rate", c_string);

  sprintf(c_string, "%d", deviceProp.multiProcessorCount);
  logger->writeValue(" MultiProcessors count", c_string);
  
  sprintf(c_string, "%d", deviceProp.concurrentKernels);
  logger->writeValue(" Concurrent Kernels", c_string);
  
  sprintf(c_string, "%d", deviceProp.kernelExecTimeoutEnabled);
  logger->writeValue(" kernel Execution Timeout Enabled", c_string);

  ///////////////////////////////////////////////////////////
  // Initialisation of the GPU device
  ///////////////////////////////////////////////////////////
  //CUT_DEVICE_INIT();
  //CHECK_CUDA_ERROR();
  if(cudaSetDevice(device)!=cudaSuccess)
  {
	  return false;
  }
  else
  {
	  /// Pude seleccionar el GPU adecudamente.
	  /// Ahora lo configuro para que se bloquee en la llamada a los kernels
	  if(cudaSetDeviceFlags(cudaDeviceBlockingSync)!=cudaSuccess)
		return false;
	  else
		return true;
  }
}