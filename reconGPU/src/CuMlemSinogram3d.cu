/**
	\file CuMlemSinogram3d.cu
	\brief Archivo que contiene la definición de la clase CuMlemSinogram3d. 
	Implementación de clase derivada de CuMlemSinogram3d, que define el algoritmo Mlem para CUDA para sinogramas3D de un cylindrical PET.

	\todo 
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2010.11.11
	\version 1.1.0
*/

#include <CuMlemSinogram3d.h>
#include "../kernels/CuMlem_kernels.cu"
//#include "../kernels/CuSiddonProjector_kernels.cu"

#ifndef __RECONGPU_GLOBALS__
#define __RECONGPU_GLOBALS__
// Memoria constante con los valores de los angulos de la proyeccion,
__device__ __constant__ float d_thetaValues_deg[MAX_PHI_VALUES];

// Memoria constante con los valores de la distancia r.
__device__ __constant__ float d_RValues_mm[MAX_R_VALUES];

// Memoria constante con los valores de la coordenada axial o z.
__device__ __constant__ float d_AxialValues_mm[MAX_Z_VALUES];

__device__ __constant__ float d_RadioScanner_mm;

__device__ __constant__ float d_AxialFov_mm;

__device__ __constant__ float d_RadioFov_mm;

__device__ __constant__ SizeImage d_imageSize;

__device__ __constant__ int d_numPixels;

__device__ __constant__ int d_numPixelsPerSlice;

__device__ __constant__ int d_numBinsSino2d;

__device__ __constant__ float d_crystalElementSize_mm;
/// Size of each sinogram's bin.
__device__ __constant__ float d_binSize_mm;
/// Depth or length og each crystal.
__device__ __constant__ float d_crystalElementLength_mm;
/// Mean depth of interaction:
__device__ __constant__ float d_meanDOI_mm; //

extern texture<float, 3, cudaReadModeElementType> texImage;  // 3D texture

extern surface<void, 3> surfImage;
  
#endif

CuMlemSinogram3d::CuMlemSinogram3d(Sinogram3D* cInputProjection, Image* cInitialEstimate, string cPathSalida, string cOutputPrefix, int cNumIterations, int cSaveIterationInterval, bool cSaveIntermediate, bool cSensitivityImageFromFile, CuProjector* cForwardprojector, CuProjector* cBackprojector) : MlemSinogram3d(cInputProjection, cInitialEstimate, cPathSalida, cOutputPrefix, cNumIterations, cSaveIterationInterval, cSaveIntermediate, cSensitivityImageFromFile, NULL, NULL)
{
  this->backprojector = cBackprojector;
  this->forwardprojector = cForwardprojector;
}

CuMlemSinogram3d::CuMlemSinogram3d(string configFilename):MlemSinogram3d(configFilename)
{
    /// Inicializo las variables con sus valores por default
    
}

bool CuMlemSinogram3d::initCuda (int device, Logger* logger)
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
  
  sprintf(c_string, "%lud", deviceProp.totalGlobalMem);
  logger->writeValue(" Total Global Memory", c_string);
  
  sprintf(c_string, "%lud", deviceProp.sharedMemPerBlock);
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
    // For this implementation, where not shared memory is used. Is better to use all the memory to cache L1:
    cudaFuncSetCacheConfig(cuSiddonProjection, cudaFuncCachePreferL1);
    return true;
  }
}

/// Método que configura los tamaños de ejecución del kernel de proyección.
void CuMlemSinogram3d::setProjectorKernelConfig(unsigned int numThreadsPerBlockX, unsigned int numThreadsPerBlockY, unsigned int numThreadsPerBlockZ)
{
  unsigned int numBlocksX = 1, numBlocksY = 1, numBlocksZ = 1;
  blockSizeProjector = dim3(numThreadsPerBlockX, numThreadsPerBlockY, numThreadsPerBlockZ);
  // Con la dimensión x de la grilla completo el sino 2d:
  numBlocksX = ceil((float)(inputProjection->getNumProj() * inputProjection->getNumR()) / blockSizeProjector.x);
  // La dimensión y, procesa cada sinograma. 
  numBlocksY = inputProjection->getNumSinograms();
    
  gridSizeProjector = dim3(numBlocksX, numBlocksY, numBlocksZ);
  
  // Con esta configuración seteo el proyector:
  forwardprojector->setKernelConfig(numThreadsPerBlockX, numThreadsPerBlockY, numThreadsPerBlockZ, numBlocksX, numBlocksY, numBlocksZ);
}

/// Método que configura los tamaños de ejecución del kernel de retroproyección.
void CuMlemSinogram3d::setBackprojectorKernelConfig(unsigned int numThreadsPerBlockX, unsigned int numThreadsPerBlockY, unsigned int numThreadsPerBlockZ)
{
  unsigned int numBlocksX = 1, numBlocksY = 1, numBlocksZ = 1;
  blockSizeBackprojector = dim3(numThreadsPerBlockX, numThreadsPerBlockY, numThreadsPerBlockZ);
  // Con la dimensión x de la grilla completo el sino 2d:
  numBlocksX = ceil((float)(inputProjection->getNumProj() * inputProjection->getNumR()) / blockSizeBackprojector.x);
  // La dimensión y procesa cada sinograma. 
  numBlocksY = inputProjection->getNumSinograms();	
  gridSizeBackprojector = dim3(numBlocksX, numBlocksY, numBlocksZ);
  
  // Con esta configuración seteo el retroproyector:
  backprojector->setKernelConfig(numThreadsPerBlockX, numThreadsPerBlockY, numThreadsPerBlockZ, numBlocksX, numBlocksY, numBlocksZ);
}

/// Método que configura los tamaños de ejecución del kernel de actualización de píxel.
void CuMlemSinogram3d::setUpdatePixelKernelConfig(unsigned int numThreadsPerBlockX, unsigned int numThreadsPerBlockY, unsigned int numThreadsPerBlockZ)
{
  unsigned int numBlocksX = 1, numBlocksY = 1, numBlocksZ = 1;
  blockSizeImageUpdate = dim3(numThreadsPerBlockX, numThreadsPerBlockY, numThreadsPerBlockZ);
  // El bloque va procesando en x, luego completo con la dimensión x de la grilla y el resto de la imagen.
  numBlocksX = ceil((float)(sizeReconImage.nPixelsX*sizeReconImage.nPixelsY) / blockSizeImageUpdate.x);
  numBlocksY = sizeReconImage.nPixelsZ;
  //numBlocksZ = sizeReconImage.nPixelsZ;
  gridSizeImageUpdate = dim3(numBlocksX, numBlocksY, numBlocksZ);  
  
}

void CuMlemSinogram3d::setProjectorKernelConfig(dim3* blockSize)
{
  // Llamo al método que recibe los componentes por separado, ya que ahí se actualiza el tamaño de grilla también:
  setProjectorKernelConfig(blockSize->x, blockSize->y, blockSize->z);
}
    
void CuMlemSinogram3d::setBackprojectorKernelConfig(dim3* blockSize)
{
  setBackprojectorKernelConfig(blockSize->x, blockSize->y, blockSize->z);
}

void CuMlemSinogram3d::setUpdatePixelKernelConfig(dim3* blockSize)
{
  setUpdatePixelKernelConfig(blockSize->x, blockSize->y, blockSize->z);	
}
    
bool CuMlemSinogram3d::InitGpuMemory(TipoProyector tipoProy)
{
  // Número total de píxeles.
  int numPixels = reconstructionImage->getPixelCount();
  // Número total de bins del sinograma:
  int numBins, numSinograms;
  numBins = inputProjection->getBinCount();
  
  // Lo mismo para el numero de sinogramas:
  numSinograms = inputProjection->getNumSinograms();
    
  float aux;
  int auxInt;
  // Pido memoria para la gpu, debo almacenar los sinogramas y las imágenes.
  // Lo hago acá y no en el proyector para mantenerme en memmoria de gpu durante toda la reconstrucción.
  checkCudaErrors(cudaMalloc((void**) &d_sensitivityImage, sizeof(float)*numPixels));
  checkCudaErrors(cudaMalloc((void**) &d_reconstructionImage, sizeof(float)*numPixels));
  checkCudaErrors(cudaMalloc((void**) &d_backprojectedImage, sizeof(float)*numPixels));
  checkCudaErrors(cudaMalloc((void**) &d_inputProjection, sizeof(float)*numBins));
  checkCudaErrors(cudaMalloc((void**) &d_estimatedProjection, sizeof(float)*numBins));
  if(enableAdditiveTerm)
    checkCudaErrors(cudaMalloc((void**) &d_additiveSinogram, sizeof(float)*numBins));
  checkCudaErrors(cudaMalloc((void**) &d_ring1, sizeof(int)*numSinograms));
  checkCudaErrors(cudaMalloc((void**) &d_ring2, sizeof(int)*numSinograms));
  // Por ahora tengo las dos, d_ring1 me da el índice de anillo, y d_ring1_mm me da directamente la coordenada axial.
  // Agregue esto porque para usar una única LOR para
  checkCudaErrors(cudaMalloc((void**) &d_ring1_mm, sizeof(int)*numSinograms));
  checkCudaErrors(cudaMalloc((void**) &d_ring2_mm, sizeof(int)*numSinograms));
  checkCudaErrors(cudaMalloc((void**) &d_likelihood, sizeof(float)));
  checkCudaErrors(cudaMemset(d_likelihood, 0,sizeof(float)));
  // Copio la iamgen inicial (esto lo hago cuando inicio la reconstrucción, así que no sería necesario):
  checkCudaErrors(cudaMemcpy(d_reconstructionImage, initialEstimate->getPixelsPtr(),sizeof(float)*numPixels,cudaMemcpyHostToDevice));
  // Pongo en cero la imágens de sensibilidad y la de retroproyección:
  checkCudaErrors(cudaMemset(d_sensitivityImage, 0,sizeof(float)*numPixels));
  checkCudaErrors(cudaMemset(d_backprojectedImage, 0,sizeof(float)*numPixels));
  // Copio el sinograma de entrada, llamo a una función porque tengo que ir recorriendo todos los sinogramas:
  CopySinogram3dHostToGpu(d_inputProjection, inputProjection);	// Es una copia de los mismos sinogramas. O sea en cpu y gpu ocupan el mismo espacio.
  // Lo mismo para el aditivo:
   if(enableAdditiveTerm)
    CopySinogram3dHostToGpu(d_additiveSinogram, additiveProjection);
  // Pongo en cero el sinograma de proyección:
  checkCudaErrors(cudaMemset(d_estimatedProjection, 0,sizeof(float)*numBins));
  
  // Además de copiar los valores de todos los bins, debo inicializar todas las constantes de reconstrucción.
  // Por un lado tengo los valores de coordenadas posibles de r, theta y z. Los mismos se copian a memoria constante de GPU (ver vectores globales al inicio de este archivo.
  float *auxPtr = inputProjection->getSegment(0)->getSinogram2D(0)->getAngPtr();
  checkCudaErrors(cudaMemcpyToSymbol(d_thetaValues_deg, auxPtr, sizeof(float)*inputProjection->getNumProj()));
  checkCudaErrors(cudaMemcpyToSymbol(d_RValues_mm, inputProjection->getSegment(0)->getSinogram2D(0)->getRPtr(), sizeof(float)*inputProjection->getNumR()));
  checkCudaErrors(cudaMemcpyToSymbol(d_AxialValues_mm, inputProjection->getAxialPtr(), sizeof(float)*inputProjection->getNumRings()));
//   checkCudaErrors(cudaMemcpyToSymbol(cuda_threads_per_block, &(blockSizeProjector.x), sizeof(unsigned int)));
//   checkCudaErrors(cudaMemcpyToSymbol(cuda_threads_per_block_update_pixel, &(blockSizeImageUpdate.x), sizeof(unsigned int)));
//   checkCudaErrors(cudaMemcpyToSymbol(cuda_nr_splitter, &NR_Splitter, sizeof(unsigned int)));
//   checkCudaErrors(cudaMemcpyToSymbol(cuda_rows_splitter, &rowSplitter, sizeof(unsigned int)));
  SizeImage size =  reconstructionImage->getSize();
  checkCudaErrors(cudaMemcpyToSymbol(d_imageSize, &size, sizeof(reconstructionImage->getSize())));
  auxInt = size.nPixelsX * size.nPixelsY;
  checkCudaErrors(cudaMemcpyToSymbol(d_numPixelsPerSlice, &auxInt, sizeof(int)));
  auxInt = auxInt * size.nPixelsZ;
  checkCudaErrors(cudaMemcpyToSymbol(d_numPixels, &auxInt, sizeof(int)));
  auxInt = inputProjection->getSegment(0)->getSinogram2D(0)->getNumProj()*inputProjection->getSegment(0)->getSinogram2D(0)->getNumR();
  checkCudaErrors(cudaMemcpyToSymbol(d_numBinsSino2d, &auxInt, sizeof(int)));
  aux = reconstructionImage->getFovRadio(); // Esto podría ser del sinograma también.
  checkCudaErrors(cudaMemcpyToSymbol(d_RadioFov_mm, &aux, sizeof(inputProjection->getRadioFov_mm())));
  aux = reconstructionImage->getFovHeight(); // Esto podría ser del sinograma.
  checkCudaErrors(cudaMemcpyToSymbol(d_AxialFov_mm, &aux, sizeof(inputProjection->getAxialFoV_mm())));

  // Para el sinograma 3d tengo que cada sino 2d puede representar varios sinogramas asociados a distintas combinaciones de anillos.
  // En la versión con CPU proceso todas las LORs, ahora solo voy a considerar la del medio, que sería la ventaja de reducir el volumen de LORs.
  // Entonces genero un array con las coordenadas de anillos de cada sinograma.
  int iSino = 0;
  int* auxRings1 = (int*)malloc(sizeof(int)*numSinograms);
  int* auxRings2 = (int*)malloc(sizeof(int)*numSinograms);
  float* auxRings1_mm = (float*)malloc(sizeof(float)*numSinograms);
  float* auxRings2_mm = (float*)malloc(sizeof(float)*numSinograms);
  float numZ;
  // Cargo la combinación de anillos, en la que busco la posición axial intermedia
  // The average position is the (axial pos for the first combin + the second for the last comb)/2. For z1 and z2. 
  for(int i = 0; i < inputProjection->getNumSegments(); i++)
  {
    for(int j = 0; j < inputProjection->getSegment(i)->getNumSinograms(); j++)
    {
      numZ = inputProjection->getSegment(i)->getSinogram2D(j)->getNumZ();
      // The ring is in fact the slice, goes from 1 to 2*numRings-1 (in c 0 to 2*numRings-2). For the real ring it would be (max+min)/2 bu since we want the slice we need to multiply by 2.
      auxRings1[iSino] = (inputProjection->getSegment(i)->getSinogram2D(j)->getRing1FromList(0)+inputProjection->getSegment(i)->getSinogram2D(j)->getRing1FromList(numZ-1));
      auxRings2[iSino] = (inputProjection->getSegment(i)->getSinogram2D(j)->getRing2FromList(0)+inputProjection->getSegment(i)->getSinogram2D(j)->getRing2FromList(numZ-1));
      // Es el promedio : cuando es par el index medio me da el índice menor pero con base 1, por eso le debo restar 1 para tener indices que inician en cero.
      auxRings1_mm[iSino] = (inputProjection->getSegment(i)->getSinogram2D(j)->getAxialValue1FromList(0) + inputProjection->getSegment(i)->getSinogram2D(j)->getAxialValue1FromList(numZ-1))/2;
      auxRings2_mm[iSino] = (inputProjection->getSegment(i)->getSinogram2D(j)->getAxialValue2FromList(0) + inputProjection->getSegment(i)->getSinogram2D(j)->getAxialValue2FromList(numZ-1))/2;
      iSino++;
    }
  }


  // Copio los índices de anillos a memoris de GPU:
  checkCudaErrors(cudaMemcpy(d_ring1, auxRings1, sizeof(int)*numSinograms, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_ring2, auxRings2, sizeof(int)*numSinograms, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_ring1_mm, auxRings1_mm, sizeof(float)*numSinograms, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_ring2_mm, auxRings2_mm, sizeof(float)*numSinograms, cudaMemcpyHostToDevice));
 
	
  /// Esto después hay que cambiarlo! Tiene que ir en la clase Michelogram!!!!!!!!!!!!!
  /// Necesito tener el dato del zfov del michelograma, que no lo tengo accesible ahora. Lo pongo a mano, pero
  /// cambiarlo de maner aurgente lo antes posible.!!!
  /// Ente el offsetZ lo calculaba en base al FOV del sinograma, ahora que fov es el de la imagen adquirida. Debo
  /// centrar dicho FOV en el FOV del sinograma y calcular el offsetZ relativo. Esto sería el valor mínimo de Z de la
  /// imagen a reconstruir. Lo puedo obtener del zFOV de la imagen o del sizePixelZ_mm.
  /// Lo mejor sería que el slice central sea el z=0, entonces no deberíamos modificar nada. Pero habría que cambiar
  /// varias funciones para que así sea. Por ahora queda así.
  //float offsetZvalue = (SCANNER_ZFOV - inputProjection->ZFOV)/2;
  //checkCudaErrors(my_cuda_error = cudaMemcpyToSymbol(OffsetZ, &offsetZvalue, sizeof(offsetZvalue)));
  
  // Datos que dependen del proyctor:
  switch(tipoProy)
  {
    case SIDDON_CYLINDRICAL_SCANNER:
      aux = ((Sinogram3DCylindricalPet*)inputProjection)->getRadioScanner_mm();
      checkCudaErrors(cudaMemcpyToSymbol(d_RadioScanner_mm, &aux, sizeof(float)));
      //checkCudaErrors(cudaMemcpy(&d_RadioScanner_mm, &aux, sizeof(aux), cudaMemcpyHostToDevice));
      break;
    case SIDDON_BACKPROJ_SURF_CYLINDRICAL_SCANNER:
      aux = ((Sinogram3DCylindricalPet*)inputProjection)->getRadioScanner_mm();
      checkCudaErrors(cudaMemcpyToSymbol(d_RadioScanner_mm, &aux, sizeof(float)));
      //checkCudaErrors(cudaMemcpy(&d_RadioScanner_mm, &aux, sizeof(aux), cudaMemcpyHostToDevice));
      break;
    case SIDDON_PROJ_TEXT_CYLINDRICAL_SCANNER:
      aux = ((Sinogram3DCylindricalPet*)inputProjection)->getRadioScanner_mm();
      checkCudaErrors(cudaMemcpyToSymbol(d_RadioScanner_mm, &aux, sizeof(float)));
      //checkCudaErrors(cudaMemcpy(&d_RadioScanner_mm, &aux, sizeof(aux), cudaMemcpyHostToDevice));
      break;
  }
  
  // Initialize texture (might be used with the some projectors):
  cudaChannelFormatDesc floatTex = cudaCreateChannelDesc<float>();
  const cudaExtent extentImageSize = make_cudaExtent(reconstructionImage->getSize().nPixelsX, reconstructionImage->getSize().nPixelsY, reconstructionImage->getSize().nPixelsZ);
  cudaMemcpy3DParms copyParams = {0};
  // set texture parameters
  texImage.normalized = false;                      // access with normalized texture coordinates
  texImage.filterMode = cudaFilterModeLinear;      // linear interpolation
  texImage.addressMode[0] = cudaAddressModeBorder;   // wrap texture coordinates
  texImage.addressMode[1] = cudaAddressModeBorder;
  texImage.addressMode[2] = cudaAddressModeBorder;
  // The image is in a texture memory:  cudaChannelFormatDesc floatTex;
  checkCudaErrors(cudaMalloc3DArray(&d_imageArray, &floatTex, extentImageSize));
  // bind array to 3D texture
  checkCudaErrors(cudaBindTextureToArray(texImage, d_imageArray, floatTex));
  
  // Libero memoria de vectores auxiliares:
  free(auxRings1);
  free(auxRings2);
  free(auxRings1_mm);
  free(auxRings2_mm);
  return true;
}

int CuMlemSinogram3d::CopySinogram3dHostToGpu(float* d_destino, Sinogram3D* h_source)
{
  // Obtengo la cantidad total de bins que tiene el sinograma 3D.
  int offset = 0;
  int numSinograms = 0;
  for(int i = 0; i < h_source->getNumSegments(); i++)
  {
    for(int j = 0; j < h_source->getSegment(i)->getNumSinograms(); j++)
    {
      checkCudaErrors(cudaMemcpy(d_destino + offset, h_source->getSegment(i)->getSinogram2D(j)->getSinogramPtr(), 
				  sizeof(float)*h_source->getSegment(i)->getSinogram2D(j)->getNumR() * h_source->getSegment(i)->getSinogram2D(j)->getNumProj(),cudaMemcpyHostToDevice));
      offset += h_source->getSegment(i)->getSinogram2D(j)->getNumR() * h_source->getSegment(i)->getSinogram2D(j)->getNumProj();
      numSinograms++;
    }
  }
  return numSinograms;
}

int CuMlemSinogram3d::CopySinogram3dGpuToHost(Sinogram3D* h_destino, float* d_source)
{
  int offset = 0;
  int numSinograms = 0;
  for(int i = 0; i < h_destino->getNumSegments(); i++)
  {
    for(int j = 0; j < h_destino->getSegment(i)->getNumSinograms(); j++)
    {
      checkCudaErrors(cudaMemcpy(h_destino->getSegment(i)->getSinogram2D(j)->getSinogramPtr(), d_source + offset, 
				  sizeof(float)*h_destino->getSegment(i)->getSinogram2D(j)->getNumR() * h_destino->getSegment(i)->getSinogram2D(j)->getNumProj(),cudaMemcpyDeviceToHost));
      offset += h_destino->getSegment(i)->getSinogram2D(j)->getNumR() * h_destino->getSegment(i)->getSinogram2D(j)->getNumProj();
      numSinograms++;
    }
  }
  return numSinograms;
}

int CuMlemSinogram3d::CopySinogram3dHostToGpuWithoutSpan(float* d_destino, Sinogram3D* h_source)
{
  int offset = 0;
  int numSinograms = 0;
  int numBinsSino2d = h_source->getSegment(0)->getSinogram2D(0)->getNumProj()*h_source->getSegment(0)->getSinogram2D(0)->getNumR();
  float* auxSino, *sino2d_fuente;
  // Pido memoria para auxSino:
  auxSino = (float*)malloc(sizeof(float)*numBinsSino2d);
  // En esta copia cada sinograma con varias combinaciones de anillos, lo debo replicar con las cuentas divididas en la cantidad de combinaciones axiales.
  for(int i = 0; i < h_source->getNumSegments(); i++)
  {
    for(int j = 0; j < h_source->getSegment(i)->getNumSinograms(); j++)
    {
      sino2d_fuente = h_source->getSegment(i)->getSinogram2D(j)->getSinogramPtr();
      for(int k = 0; k < h_source->getSegment(i)->getSinogram2D(j)->getNumZ(); k++)
      {
	// En auxsino, guardo el sinograma a copiar. Esto es necesario para normalizar por la combinación de anillos que representa dicho sinograma:
	for(int l = 0; l < numBinsSino2d; l++)
	{
	  auxSino[l] = sino2d_fuente[l] / h_source->getSegment(i)->getSinogram2D(j)->getNumZ();
	}
	checkCudaErrors(cudaMemcpy(d_destino + offset, auxSino, 
				    sizeof(float)*numBinsSino2d,cudaMemcpyHostToDevice));
	offset += numBinsSino2d;
	numSinograms++;
      }
    }
  }
  // Libero memoria:
  free(auxSino);
  return numSinograms;
}

int CuMlemSinogram3d::CopySinogram3dGpuWithoutSpanToHost(Sinogram3D* h_destino, float* d_source)
{
  int offset = 0;
  int numSinograms = 0;
  int numBinsSino2d = h_destino->getSegment(0)->getSinogram2D(0)->getNumProj()*h_destino->getSegment(0)->getSinogram2D(0)->getNumR();
  float* sino2d;	// Puntero que maneja el sino 2d del sinograma 3d.
  float* auxSino;
  // Pido memoria para auxSino:
  auxSino = (float*)malloc(sizeof(float)*numBinsSino2d);
  // Leno en cero porque voy acumulando:
  h_destino->FillConstant(0);
  // Ahora mientras copio debo hacer la compresión axial. Esto lo voy a hacer básicamente sumando los mismos sinogramas
  for(int i = 0; i < h_destino->getNumSegments(); i++)
  {
    for(int j = 0; j < h_destino->getSegment(i)->getNumSinograms(); j++)
    {
      sino2d = h_destino->getSegment(i)->getSinogram2D(j)->getSinogramPtr();
      
      for(int k = 0; k < h_destino->getSegment(i)->getSinogram2D(j)->getNumZ(); k++)
      {
	// Copio el sinograma en auxSino:
	checkCudaErrors(cudaMemcpy(auxSino, d_source + offset, 
				    sizeof(float)*numBinsSino2d,cudaMemcpyDeviceToHost));
	// En auxsino tengo el sinograma correspondiente a una combinación de anillos, que la voy a ir sumando al sinograma en host que tiene
	// compresión axial mediante cierto span.
	for(int l = 0; l < h_destino->getSegment(i)->getSinogram2D(j)->getNumProj()*h_destino->getSegment(i)->getSinogram2D(j)->getNumR(); l++)
	{
	  sino2d[l] += auxSino[l];
	}
	
	offset += numBinsSino2d; // El offset corresponde a los sinos en gpu.
      }
      numSinograms++; // número de sinos en cpu con compresión axial.
    }
  }
  // Libero auxsino:
  free(auxSino);
  return numSinograms;
}

void CuMlemSinogram3d::CopyReconstructedImageGpuToHost()
{
  float* aux = reconstructionImage->getPixelsPtr();
  checkCudaErrors(cudaMemcpy(aux, d_reconstructionImage, sizeof(float)*reconstructionImage->getPixelCount(),cudaMemcpyDeviceToHost)); 
}

void CuMlemSinogram3d::CopyReconstructedImageHostToGpu()
{
  float* aux = reconstructionImage->getPixelsPtr();
  checkCudaErrors(cudaMemcpy(d_reconstructionImage, aux, sizeof(float)*reconstructionImage->getPixelCount(),cudaMemcpyHostToDevice)); 
}

// Método de reconstrucción que no se le indica el índice de GPU, incializa la GPU 0 por defecto.
bool CuMlemSinogram3d::Reconstruct(TipoProyector tipoProy)
{
  Reconstruct(tipoProy, 0);
  return true;
}

/// Método público que realiza la reconstrucción en base a los parámetros pasados al objeto Mlem instanciado
bool CuMlemSinogram3d::Reconstruct(TipoProyector tipoProy, int indexGpu)
{
  string outputFilename;	// String para los nombres de los archivos de salida.
  // Instancio proyector:
  
  
  /// Hago el log de la reconstrucción:
  Logger* logger = new Logger(logFileName);
  // INICALIZACIÓN DE GPU 
  if(!initCuda (0, logger))
  {
    return false;
  }
  // Inicializo memoria de GPU:
  this->InitGpuMemory(tipoProy);
  
  // Sigo con variables necesarias durante la reconstrucción:
  /// Tamaño de la imagen:
  SizeImage sizeImage = reconstructionImage->getSize();
  /// Proyección auxiliar, donde guardo el sinograma proyectado:
  Sinogram3D* estimatedProjection = inputProjection->Copy();
  estimatedProjection->FillConstant(0);
  /// Imagen donde guardo la backprojection.
  Image* backprojectedImage = new Image(reconstructionImage->getSize());
  /// Puntero a la imagen.
  float* ptrPixels = reconstructionImage->getPixelsPtr();
  /// Puntero a la sensitivity image.
  float* ptrSensitivityPixels = sensitivityImage->getPixelsPtr();
  /// Puntero a la sensitivity image.
  float* ptrBackprojectedPixels = backprojectedImage->getPixelsPtr();
  /// Puntero del array con los tiempos de reconstrucción por iteración.
  float* timesIteration_mseg;
  /// Puntero del array con los tiempos de backprojection por iteración.
  float* timesBackprojection_mseg;
  /// Puntero del array con los tiempos de forwardprojection por iteración.
  float* timesForwardprojection_mseg;
  /// Puntero del array con los tiempos de pixel update por iteración.
  float* timesPixelUpdate_mseg;
  /// String de c para utlizar en los mensajes de logueo.
  char c_string[512];
  /// Pido memoria para los arrays, que deben tener tantos elementos como iteraciones:
  timesIteration_mseg = (float*)malloc(sizeof(float)*this->numIterations);
  timesBackprojection_mseg = (float*)malloc(sizeof(float)*this->numIterations);
  timesForwardprojection_mseg = (float*)malloc(sizeof(float)*this->numIterations);
  timesPixelUpdate_mseg = (float*)malloc(sizeof(float)*this->numIterations);
  /// El vector de likelihood puede haber estado alocado previamente por lo que eso realloc. Tiene
  /// un elemento más porque el likelihood es previo a la actualización de la imagen, o sea que inicia
  /// con el initialEstimate y termina con la imagen reconstruida.
  if(this->likelihoodValues == NULL)
  {
    /// No había sido alocado previamente así que utilizo malloc.
    this->likelihoodValues = (float*)malloc(sizeof(float)*(this->numIterations +1));
  }
  else
  {
    /// Ya había sido alocado previamente, lo realoco.
    this->likelihoodValues = (float*)realloc(this->likelihoodValues, sizeof(float)*(this->numIterations + 1));
  }
  /// Me fijo si la sensitivity image la tengo que cargar desde archivo o calcularla
  if(sensitivityImageFromFile)
  {
    /// Leo el sensitivity volume desde el archivo
    sensitivityImage->readFromInterfile((char*) sensitivityFilename.c_str());
    ptrSensitivityPixels = sensitivityImage->getPixelsPtr();
    updateUpdateThreshold();
  }
  else
  {
    /// Calculo el sensitivity volume
    if(computeSensitivity(tipoProy)==false)
    {
      strError = "Error al calcular la sensitivity Image.";
      return false;
    }
    // La guardo en disco.
    string sensitivityFileName = outputFilenamePrefix;
    sensitivityFileName.append("_sensitivity");
    sensitivityImage->writeInterfile((char*)sensitivityFileName.c_str());
  }
  /// Inicializo el volumen a reconstruir con la imagen del initial estimate:
  reconstructionImage = new Image(initialEstimate);
  ptrPixels = reconstructionImage->getPixelsPtr();
  // Copio la imagen a device.
  CopyReconstructedImageHostToGpu();
  
  /// Escribo el título y luego los distintos parámetros de la reconstrucción:
  logger->writeLine("######## CUDA ML-EM Reconstruction #########");
  logger->writeValue("Name", this->outputFilenamePrefix);
  logger->writeValue("Type", "ML-EM");
  sprintf(c_string, "%d", this->numIterations);
  logger->writeValue("Iterations", c_string);
  logger->writeValue("Input Projections", "3D Sinogram");
  sprintf(c_string, "%d[r] x %d[ang]", inputProjection->getNumR(), inputProjection->getNumProj());
  logger->writeValue("Size of Sinogram2D",c_string);
  sprintf(c_string, "%d", inputProjection->getNumRings());
  logger->writeValue("Rings", c_string);
  sprintf(c_string, "%d", inputProjection->getNumSegments());
  logger->writeValue("Segments", c_string);
  sprintf(c_string, "%d", sizeReconImage.nDimensions);
  logger->writeValue("Image Dimensions", c_string);
  sprintf(c_string, "%d[x] x %d[y] x %d[z]", this->sizeReconImage.nPixelsX, this->sizeReconImage.nPixelsY, this->sizeReconImage.nPixelsZ);
  logger->writeValue("Image Size", c_string);
  sprintf(c_string, "Bloques %d[x] x %d[y] x %d[z]. Grilla %d[x] x %d[y] x %d[z]", this->blockSizeProjector.x, this->blockSizeProjector.y, this->blockSizeProjector.z,
    this->gridSizeProjector.x, this->gridSizeProjector.y, this->gridSizeProjector.z);
  logger->writeValue("Projection Kernel Config", c_string);
  sprintf(c_string, "Bloques %d[x] x %d[y] x %d[z]. Grilla %d[x] x %d[y] x %d[z]", this->blockSizeBackprojector.x, this->blockSizeBackprojector.y, this->blockSizeBackprojector.z,
    this->gridSizeBackprojector.x, this->gridSizeBackprojector.y, this->gridSizeBackprojector.z);
  logger->writeValue("Backprojection Kernel Config", c_string);
  sprintf(c_string, "Bloques %d[x] x %d[y] x %d[z]. Grilla %d[x] x %d[y] x %d[z]", this->blockSizeImageUpdate.x, this->blockSizeImageUpdate.y, this->blockSizeImageUpdate.z,
    this->gridSizeImageUpdate.x, this->gridSizeImageUpdate.y, this->gridSizeImageUpdate.z);
  logger->writeValue("Pixel Update Kernel Config", c_string);
  // También se realiza un registro de los tiempos de ejecución:
  clock_t initialClock = clock();
  //Start with the iteration
  printf("Iniciando Reconstrucción...\n");
  /// Arranco con el log de los resultados:
  strcpy(c_string, "_______RECONSTRUCCION_______");
  logger->writeLine(c_string, strlen(c_string));
  /// Voy generando mensajes con los archivos creados en el log de salida:
  int nPixels = reconstructionImage->getPixelCount();
  int nBins = estimatedProjection->getBinCount();
  int nBinsSino2d = estimatedProjection->getSegment(0)->getSinogram2D(0)->getNumProj()*estimatedProjection->getSegment(0)->getSinogram2D(0)->getNumR();
  
  for(unsigned int t = 0; t < this->numIterations; t++)
  {
    clock_t initialClockIteration = clock();
    printf("Iteración Nº: %d\n", t);
    /// Pongo en cero la proyección estimada, y hago la backprojection.
    checkCudaErrors(cudaMemset(d_estimatedProjection, 0,sizeof(float)*nBins));
    /// Proyección de la imagen:
    switch(tipoProy)
    {
      case SIDDON_PROJ_TEXT_CYLINDRICAL_SCANNER: // This siddon implementation has only projection, so it uses the standard backprojection.
	CopyDevImageToTexture(d_reconstructionImage, sensitivityImage->getSize()); // Copy the reconstruction iamge to texture.
	forwardprojector->Project(d_reconstructionImage, d_estimatedProjection, d_ring1_mm, d_ring2_mm, reconstructionImage, (Sinogram3DCylindricalPet*)inputProjection, false);
	break;
      case SIDDON_CYLINDRICAL_SCANNER:
	forwardprojector->Project(d_reconstructionImage, d_estimatedProjection, d_ring1_mm, d_ring2_mm, reconstructionImage, (Sinogram3DCylindricalPet*)inputProjection, false);
	break;
    }
    // The additive term in the forward model (the multiplicative is only take into account in the sensitivity image,
    // so the additive term need to be dividived by the multipicative factors previously):
    if(enableAdditiveTerm)
      addSinograms(d_estimatedProjection, d_additiveSinogram, nBinsSino2d, nBins);
    clock_t finalClockProjection = clock();
    
    /// Guardo el likelihood (Siempre va una iteración atrás, ya que el likelihhod se calcula a partir de la proyección
    /// estimada, que es el primer paso del algoritmo). Se lo calculo al sinograma
    /// proyectado, respecto del de entrada.
    this->likelihoodValues[t] = this->getLikelihoodValue();
    /// Si quiero guardar la proyección intermedia, lo hago acá, porque luego en la backprojection se modifica para hacer el cociente entre entrada y estimada:
    if(saveIntermediateProjectionAndBackprojectedImage)
    {
      CopySinogram3dGpuToHost(estimatedProjection, d_estimatedProjection);
      sprintf(c_string, "%s_projection_iter_%d", outputFilenamePrefix.c_str(), t); /// La extensión se le agrega en write interfile.
      outputFilename.assign(c_string);
      estimatedProjection->writeInterfile((char*)outputFilename.c_str());
    }
    /// Pongo en cero la imagen de corrección, y hago la backprojection.
    checkCudaErrors(cudaMemset(d_backprojectedImage, 0,sizeof(float)*nPixels));
    switch(tipoProy)
    {
      case SIDDON_PROJ_TEXT_CYLINDRICAL_SCANNER: // This siddon implementation has only projection, so it uses the standard backprojection.
      case SIDDON_CYLINDRICAL_SCANNER:
	backprojector->DivideAndBackproject(d_inputProjection, d_estimatedProjection, d_backprojectedImage, d_ring1_mm, d_ring2_mm, (Sinogram3DCylindricalPet*)inputProjection, backprojectedImage, false);
	break;
    }
    if(saveIntermediateProjectionAndBackprojectedImage)
    {
      // Copio la imagen en gpu a cpu:
      checkCudaErrors(cudaMemcpy(backprojectedImage->getPixelsPtr(), d_backprojectedImage, sizeof(float)*reconstructionImage->getPixelCount(),cudaMemcpyDeviceToHost)); 
      sprintf(c_string, "%s_backprojection_iter_%d", outputFilenamePrefix.c_str(), t); /// La extensión se le agrega en write interfile.
      outputFilename.assign(c_string);
      backprojectedImage->writeInterfile((char*)outputFilename.c_str());
    }
    clock_t finalClockBackprojection = clock();
    /// Actualización del Pixel
    this->updatePixelValue();
    /// Verifico
    if(saveIterationInterval != 0)
    {
      if((t%saveIterationInterval)==0)
      {
	// Primero tengo que obtener la memoria de GPU:
	CopyReconstructedImageGpuToHost();
	sprintf(c_string, "%s_iter_%d", outputFilenamePrefix.c_str(), t); /// La extensión se le agrega en write interfile.
	outputFilename.assign(c_string);
	reconstructionImage->writeInterfile((char*)outputFilename.c_str());
	/// Termino con el log de los resultados:
	sprintf(c_string, "Imagen de iteración %d guardada en: %s", t, outputFilename.c_str());
	logger->writeLine(c_string);
      }
    }
    clock_t finalClockIteration = clock();
    /// Cargo los tiempos:
    timesIteration_mseg[t] = (float)(finalClockIteration-initialClockIteration)*1000/(float)CLOCKS_PER_SEC;
    timesBackprojection_mseg[t] = (float)(finalClockBackprojection-finalClockProjection)*1000/(float)CLOCKS_PER_SEC;
    timesForwardprojection_mseg[t] = (float)(finalClockProjection-initialClockIteration)*1000/(float)CLOCKS_PER_SEC;
    timesPixelUpdate_mseg[t] = (float)(finalClockIteration-finalClockBackprojection)*1000/(float)CLOCKS_PER_SEC;

  }
  
  // Copio resultado:
  CopyReconstructedImageGpuToHost();
  clock_t finalClock = clock();
  sprintf(c_string, "%s_final", outputFilenamePrefix.c_str()); /// La extensión se le agrega en write interfile.
  outputFilename.assign(c_string);
  reconstructionImage->writeInterfile((char*)outputFilename.c_str());
  /// Termino con el log de los resultados:
  sprintf(c_string, "Imagen final guardada en: %s", outputFilename.c_str());
  logger->writeLine(c_string);
  /// Calculo la proyección de la última imagen para poder calcular el likelihood final:
  switch(tipoProy)
  {
    case SIDDON_CYLINDRICAL_SCANNER:
      forwardprojector->Project(d_reconstructionImage, d_estimatedProjection, d_ring1_mm, d_ring2_mm, reconstructionImage, (Sinogram3DCylindricalPet*)inputProjection, false);
      break;
  }
  this->likelihoodValues[this->numIterations] = getLikelihoodValue();

  float tiempoTotal = (float)(finalClock - initialClock)*1000/(float)CLOCKS_PER_SEC;
  /// Termino con el log de los resultados:
  strcpy(c_string, "_______RESULTADOS DE RECONSTRUCCION_______");
  logger->writeLine(c_string, strlen(c_string));
  sprintf(c_string, "%f", tiempoTotal);
  logger->writeValue("Tiempo Total de Reconstrucción:", c_string);
  /// Ahora guardo los tiempos por iteración y por etapa, en fila de valores.
  strcpy(c_string, "Tiempos de Reconstrucción por Iteración [mseg]");
  logger->writeLine(c_string, strlen(c_string));
  logger->writeRowOfNumbers(timesIteration_mseg, this->numIterations);
  strcpy(c_string, "Tiempos de Forwardprojection por Iteración [mseg]");
  logger->writeLine(c_string, strlen(c_string));
  logger->writeRowOfNumbers(timesForwardprojection_mseg, this->numIterations);
  strcpy(c_string, "Tiempos de Backwardprojection por Iteración [mseg]");
  logger->writeLine(c_string, strlen(c_string));
  logger->writeRowOfNumbers(timesBackprojection_mseg, this->numIterations);
  strcpy(c_string, "Tiempos de UpdatePixel por Iteración [mseg]");
  logger->writeLine(c_string, strlen(c_string));
  logger->writeRowOfNumbers(timesPixelUpdate_mseg, this->numIterations);
  /// Por último registro los valores de likelihood:
  strcpy(c_string, "Likelihood por Iteración:");
  logger->writeLine(c_string, strlen(c_string));
  logger->writeRowOfNumbers(this->likelihoodValues, this->numIterations + 1);

  /// Libero la memoria de los arrays:
  free(timesIteration_mseg);
  free(timesBackprojection_mseg);
  free(timesForwardprojection_mseg);
  free(timesPixelUpdate_mseg);
  return true;
}


float CuMlemSinogram3d::getLikelihoodValue()
{
  float likelihood;
  checkCudaErrors(cudaMemset(d_likelihood, 0,sizeof(float)));
  cuGetLikelihoodValue<<<gridSizeProjector, blockSizeProjector>>>(d_estimatedProjection, d_inputProjection, d_likelihood, inputProjection->getNumR(), inputProjection->getNumProj(), inputProjection->getNumRings(), inputProjection->getNumSinograms());
  /// Sincronización de todos los threads.
  cudaThreadSynchronize();
  checkCudaErrors(cudaMemcpy(&likelihood, d_likelihood,sizeof(float),cudaMemcpyDeviceToHost));
  return likelihood;
  
}

bool CuMlemSinogram3d::updatePixelValue()
{
  // Llamo al kernel que actualiza el pixel.
  cuUpdatePixelValue<<<gridSizeImageUpdate, blockSizeImageUpdate>>>(d_reconstructionImage, d_backprojectedImage, d_sensitivityImage, reconstructionImage->getSize(), updateThreshold);
  cudaThreadSynchronize();
  return true;
}

bool CuMlemSinogram3d::addSinograms(float* d_inputOuputSino1, float* d_inputSino2, int numBinsPerSlice,  int numElements)
{
  // Llamo al kernel que actualiza el pixel.
  cuAddVectors<<<gridSizeProjector, blockSizeProjector>>>(d_inputOuputSino1, d_inputSino2, numBinsPerSlice, numElements);
  cudaThreadSynchronize();
  return true;
}


bool CuMlemSinogram3d::computeSensitivity(TipoProyector tipoProy)
{
  /// Creo un Sinograma ·D igual que el de entrada.
  Sinogram3D* constantSinogram3D;
  /// With normalization use the norm sinogram if not a constant sinogram:
  if (enableMultiplicativeTerm)
    constantSinogram3D = multiplicativeProjection->Copy();
  else
  {
    constantSinogram3D = inputProjection->Copy();
    constantSinogram3D->FillConstant(1);
  }
  /// Copio a gpu:
  CopySinogram3dHostToGpu(d_estimatedProjection, constantSinogram3D);
  /// Por último hago la backprojection
  switch(tipoProy)
  {
    case SIDDON_PROJ_TEXT_CYLINDRICAL_SCANNER: // This siddon implementation has only projection, so it uses the standard backprojection.
    case SIDDON_CYLINDRICAL_SCANNER:
      backprojector->Backproject(d_estimatedProjection, d_sensitivityImage, d_ring1_mm, d_ring2_mm, (Sinogram3DCylindricalPet*)constantSinogram3D, reconstructionImage, false);
      // Copio la memoria de gpu a cpu, así se puede actualizar el umbral:
      checkCudaErrors(cudaMemcpy(sensitivityImage->getPixelsPtr(), d_sensitivityImage,sizeof(float)*sensitivityImage->getPixelCount(),cudaMemcpyDeviceToHost));
      break;
    case SIDDON_BACKPROJ_SURF_CYLINDRICAL_SCANNER:
      CopyDevImageToTexture(d_sensitivityImage, sensitivityImage->getSize());
      backprojector->Backproject(d_estimatedProjection, d_sensitivityImage, d_ring1_mm, d_ring2_mm, (Sinogram3DCylindricalPet*)constantSinogram3D, reconstructionImage, false);
      CopyTextureToDevtImage(d_sensitivityImage, sensitivityImage->getSize());
      CopyTextureToHostImage(sensitivityImage);
      break;
  }
  // Umbral para la actualización de píxel:
  updateUpdateThreshold();
  return true;
}

bool CuMlemSinogram3d::CopyHostImageToTexture(Image* image)
{ 
  const cudaExtent extentImageSize = make_cudaExtent(image->getSize().nPixelsX, image->getSize().nPixelsY, image->getSize().nPixelsZ);
  cudaMemcpy3DParms copyParams = {0};
  // copy data to 3D array
  copyParams.srcPtr   = make_cudaPitchedPtr((void *)image->getPixelsPtr(), extentImageSize.width*sizeof(float), extentImageSize.width, extentImageSize.height);
  copyParams.dstArray = d_imageArray;
  copyParams.extent   = extentImageSize;
  copyParams.kind     = cudaMemcpyHostToDevice;
  checkCudaErrors(cudaMemcpy3D(&copyParams));
  return true;
}

bool CuMlemSinogram3d::CopyTextureToHostImage(Image* image)
{ 
  cudaMemcpy3DParms copyParams = {0};
  const cudaExtent extentImageSize = make_cudaExtent(image->getSize().nPixelsX, image->getSize().nPixelsY, image->getSize().nPixelsZ);
  copyParams.srcArray   = d_imageArray;
  copyParams.dstPtr = make_cudaPitchedPtr((void *)image->getPixelsPtr(), extentImageSize.width*sizeof(float), extentImageSize.width, extentImageSize.height);
  copyParams.extent   = extentImageSize;
  copyParams.kind     = cudaMemcpyDeviceToHost;
  checkCudaErrors(cudaMemcpy3D(&copyParams));
  return true;
} 

bool CuMlemSinogram3d::CopyDevImageToTexture(float* d_image, SizeImage imageSize)
{ 
  const cudaExtent extentImageSize = make_cudaExtent(imageSize.nPixelsX, imageSize.nPixelsY, imageSize.nPixelsZ);
  cudaMemcpy3DParms copyParams = {0};
  // copy data to 3D array
  copyParams.srcPtr   = make_cudaPitchedPtr((void *)d_image, extentImageSize.width*sizeof(float), extentImageSize.width, extentImageSize.height);
  copyParams.dstArray = d_imageArray;
  copyParams.extent   = extentImageSize;
  copyParams.kind     = cudaMemcpyDeviceToDevice;
  checkCudaErrors(cudaMemcpy3D(&copyParams));
  return true;
}

bool CuMlemSinogram3d::CopyTextureToDevtImage(float* d_image, SizeImage imageSize)
{ 
  cudaMemcpy3DParms copyParams = {0};
  const cudaExtent extentImageSize = make_cudaExtent(imageSize.nPixelsX, imageSize.nPixelsY, imageSize.nPixelsZ);
  copyParams.srcArray   = d_imageArray;
  copyParams.dstPtr = make_cudaPitchedPtr((void *)d_image, extentImageSize.width*sizeof(float), extentImageSize.width, extentImageSize.height);
  copyParams.extent   = extentImageSize;
  copyParams.kind     = cudaMemcpyDeviceToDevice;
  checkCudaErrors(cudaMemcpy3D(&copyParams));
  return true;
} 