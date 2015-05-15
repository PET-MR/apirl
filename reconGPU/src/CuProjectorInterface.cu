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

#include <CuProjectorInterface.h>
//#include "../kernels/CuSiddonProjector_kernels.cu"

/* IMPORTANT: THIS IFDEF IS TO AVOID REEDEFINITION OF THIS GLOBAL VARIABLES IN RECONGPU LIB. THEY ARE SAHED BETWEEN PROJECTORS, MLEM AND OSEM. THIS IS
 TO AVOIDA MKING DIFFERENT PROJECTS FOR EACH OF THEM.*/
/*#ifndef __RECONGPU_GLOBALS__
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
#else*/
  /// Memoria Estática de GPU definido en Otra clase (Debería ser en la de reconstrucción: CuMlem, CuOsem, etc)
  // Memoria constante con los valores de los angulos de la proyeccion,
  extern  __device__ __constant__ float d_thetaValues_deg[MAX_PHI_VALUES];

  // Memoria constante con los valores de la distancia r.
  extern __device__ __constant__ float d_RValues_mm[MAX_R_VALUES];

  // Memoria constante con los valores de la coordenada axial o z.
  extern __device__ __constant__ float d_AxialValues_mm[MAX_Z_VALUES];

  // Memoria constante con el radio del scanner (solo para scanner cilíndricos).
  extern __device__ __constant__ float d_RadioScanner_mm;

  extern __device__ __constant__ float d_AxialFov_mm;

  extern __device__ __constant__ float d_RadioFov_mm;

  extern __device__ __constant__ SizeImage d_imageSize;
//#endif


CuProjectorInterface::CuProjectorInterface(CuProjector* cuProjector)
{
  this->projector = cuProjector;
  gpuId = 0;
  typeOfProjector = SIDDON_CYLINDRICAL_SCANNER;	// The only one available at the moment.
}


bool CuProjectorInterface::initCuda (int device)
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
  
  printf("______GPU PROPERTIES_____");
  printf(" Nombre: %s", deviceProp.name);
  
  printf(" Cuda Driver Version %d", driverVersion);
  printf(" Cuda Runtime Version %d", runtimeVersion);
  printf(" Compute Mode %d",  deviceProp.computeMode);
  printf(" Total Global Memory %lud", deviceProp.totalGlobalMem);
  printf(" Shared Memory per Block %lud", deviceProp.sharedMemPerBlock);
  printf(" Register pero Block %d", deviceProp.regsPerBlock);
  printf(" Warp Size %d", deviceProp.warpSize);
  printf(" Max Threads Per Block %d", deviceProp.maxThreadsPerBlock);
  printf(" Max Threads Dimension %dx%dx%d", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
  printf(" Max Grid Size %dx%dx%d", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
  printf(" Max Threads Per Block %d", deviceProp.maxThreadsPerBlock);
  printf(" Clock Rate %d", deviceProp.clockRate);
  printf(" MultiProcessors count %d", deviceProp.multiProcessorCount);
  printf(" Concurrent Kernels %d", deviceProp.concurrentKernels);
  printf(" kernel Execution Timeout Enabled %d", deviceProp.kernelExecTimeoutEnabled);
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

/// Método que configura los tamaños de ejecución del kernel de proyección.
void CuProjectorInterface::setProjectorKernelConfig(unsigned int numThreadsPerBlockX, unsigned int numThreadsPerBlockY, unsigned int numThreadsPerBlockZ, Sinogram3D* sinogram)
{
  unsigned int numBlocksX = 1, numBlocksY = 1, numBlocksZ = 1;
  blockSizeProjector = dim3(numThreadsPerBlockX, numThreadsPerBlockY, numThreadsPerBlockZ);
  // Con la dimensión x de la grilla completo el sino 2d:
  numBlocksX = ceil((float)(sinogram->getNumProj() * sinogram->getNumR()) / blockSizeProjector.x);
  // La dimensión y, procesa cada sinograma. 
  numBlocksY = sinogram->getNumSinograms();
    
  gridSizeProjector = dim3(numBlocksX, numBlocksY, numBlocksZ);
  
  // Con esta configuración seteo el proyector:
  projector->setKernelConfig(numThreadsPerBlockX, numThreadsPerBlockY, numThreadsPerBlockZ, numBlocksX, numBlocksY, numBlocksZ);
}


void CuProjectorInterface::setProjectorKernelConfig(dim3* blockSize, Sinogram3D* sinogram)
{
  // Llamo al método que recibe los componentes por separado, ya que ahí se actualiza el tamaño de grilla también:
  setProjectorKernelConfig(blockSize->x, blockSize->y, blockSize->z, sinogram);
}
    
    
bool CuProjectorInterface::InitGpuMemory(Sinogram3D* sinogram, Image* image, TipoProyector tipoProy)
{
  // Número total de píxeles.
  int numPixels = image->getPixelCount();
  // Número total de bins del sinograma:
  int numBins, numSinograms;
  numBins = sinogram->getBinCount();
  
  // Lo mismo para el numero de sinogramas:
  numSinograms = sinogram->getNumSinograms();
    
  float aux;
  // Pido memoria para la gpu, debo almacenar el sinograma y la imagen.
  checkCudaErrors(cudaMalloc((void**) &d_image, sizeof(float)*numPixels));
  checkCudaErrors(cudaMalloc((void**) &d_projection, sizeof(float)*numBins));
  checkCudaErrors(cudaMalloc((void**) &d_ring1, sizeof(int)*numSinograms));
  checkCudaErrors(cudaMalloc((void**) &d_ring2, sizeof(int)*numSinograms));
  // Por ahora tengo las dos, d_ring1 me da el índice de anillo, y d_ring1_mm me da directamente la coordenada axial.
  // Agregue esto porque para usar una única LOR para
  checkCudaErrors(cudaMalloc((void**) &d_ring1_mm, sizeof(int)*numSinograms));
  checkCudaErrors(cudaMalloc((void**) &d_ring2_mm, sizeof(int)*numSinograms));
  // Copio la iamgen inicial (esto lo hago cuando inicio la reconstrucción, así que no sería necesario):
  checkCudaErrors(cudaMemcpy(d_image, image->getPixelsPtr(),sizeof(float)*numPixels,cudaMemcpyHostToDevice));
  // Copio el sinograma de entrada, llamo a una función porque tengo que ir recorriendo todos los sinogramas:
  CopySinogram3dHostToGpu(d_projection, sinogram);	// Es una copia de los mismos sinogramas. O sea en cpu y gpu ocupan el mismo espacio.
  // Pongo en cero el sinograma de proyección:
  checkCudaErrors(cudaMemset(d_projection, 0,sizeof(float)*numBins));
  
  // Además de copiar los valores de todos los bins, debo inicializar todas las constantes de reconstrucción.
  // Por un lado tengo los valores de coordenadas posibles de r, theta y z. Los mismos se copian a memoria constante de GPU (ver vectores globales al inicio de este archivo.
  float *auxPtr = sinogram->getSegment(0)->getSinogram2D(0)->getAngPtr();
  checkCudaErrors(cudaMemcpyToSymbol(d_thetaValues_deg, auxPtr, sizeof(float)*sinogram->getNumProj()));
  checkCudaErrors(cudaMemcpyToSymbol(d_RValues_mm, sinogram->getSegment(0)->getSinogram2D(0)->getRPtr(), sizeof(float)*sinogram->getNumR()));
  checkCudaErrors(cudaMemcpyToSymbol(d_AxialValues_mm, sinogram->getAxialPtr(), sizeof(float)*sinogram->getNumRings()));
//   checkCudaErrors(cudaMemcpyToSymbol(cuda_threads_per_block, &(blockSizeProjector.x), sizeof(unsigned int)));
//   checkCudaErrors(cudaMemcpyToSymbol(cuda_threads_per_block_update_pixel, &(blockSizeImageUpdate.x), sizeof(unsigned int)));
//   checkCudaErrors(cudaMemcpyToSymbol(cuda_nr_splitter, &NR_Splitter, sizeof(unsigned int)));
//   checkCudaErrors(cudaMemcpyToSymbol(cuda_rows_splitter, &rowSplitter, sizeof(unsigned int)));
  SizeImage size =  image->getSize();
  checkCudaErrors(cudaMemcpyToSymbol(d_imageSize, &size, sizeof(image->getSize())));
  aux = image->getFovRadio(); // Esto podría ser del sinograma también.
  checkCudaErrors(cudaMemcpyToSymbol(d_RadioFov_mm, &aux, sizeof(sinogram->getRadioFov_mm())));
  aux = image->getFovHeight(); // Esto podría ser del sinograma.
  checkCudaErrors(cudaMemcpyToSymbol(d_AxialFov_mm, &aux, sizeof(sinogram->getAxialFoV_mm())));

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
  for(int i = 0; i < sinogram->getNumSegments(); i++)
  {
    for(int j = 0; j < sinogram->getSegment(i)->getNumSinograms(); j++)
    {
      numZ = sinogram->getSegment(i)->getSinogram2D(j)->getNumZ();
      // The ring is in fact the slice, goes from 1 to 2*numRings-1 (in c 0 to 2*numRings-2). For the real ring it would be (max+min)/2 bu since we want the slice we need to multiply by 2.
      auxRings1[iSino] = (sinogram->getSegment(i)->getSinogram2D(j)->getRing1FromList(0)+sinogram->getSegment(i)->getSinogram2D(j)->getRing1FromList(numZ-1));
      auxRings2[iSino] = (sinogram->getSegment(i)->getSinogram2D(j)->getRing2FromList(0)+sinogram->getSegment(i)->getSinogram2D(j)->getRing2FromList(numZ-1));
      // Es el promedio : cuando es par el index medio me da el índice menor pero con base 1, por eso le debo restar 1 para tener indices que inician en cero.
      auxRings1_mm[iSino] = (sinogram->getSegment(i)->getSinogram2D(j)->getAxialValue1FromList(0) + sinogram->getSegment(i)->getSinogram2D(j)->getAxialValue1FromList(numZ-1))/2;
      auxRings2_mm[iSino] = (sinogram->getSegment(i)->getSinogram2D(j)->getAxialValue2FromList(0) + sinogram->getSegment(i)->getSinogram2D(j)->getAxialValue2FromList(numZ-1))/2;
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
      aux = ((Sinogram3DCylindricalPet*)sinogram)->getRadioScanner_mm();
      checkCudaErrors(cudaMemcpyToSymbol(d_RadioScanner_mm, &aux, sizeof(aux)));
      //checkCudaErrors(cudaMemcpy(&d_RadioScanner_mm, &aux, sizeof(aux), cudaMemcpyHostToDevice));
      break;
  }
  
  // Libero memoria de vectores auxiliares:
  free(auxRings1);
  free(auxRings2);
  free(auxRings1_mm);
  free(auxRings2_mm);
  return true;
}

int CuProjectorInterface::CopySinogram3dHostToGpu(float* d_destino, Sinogram3D* h_source)
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

int CuProjectorInterface::CopySinogram3dGpuToHost(Sinogram3D* h_destino, float* d_source)
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



bool CuProjectorInterface::Backproject (Sinogram3D* inputSinogram, Image* outputImage)
{
  clock_t initTime, finalTime;
  // Número total de píxeles.
  int numPixels = outputImage->getPixelCount();
  // Número total de bins del sinograma:
  int numBins, numSinograms;
  numBins = inputSinogram->getBinCount();
  printf("######## BACKPROJECTION #########");
  // INICALIZACIÓN DE GPU 
  if(!initCuda (gpuId))
  {
    return false;
  }
  printf(" Kernel Configuration: Block %d[x] x %d[y] x %d[z]. Grid %d[x] x %d[y] x %d[z]\n", this->blockSizeProjector.x, this->blockSizeProjector.y, this->blockSizeProjector.z,
    this->gridSizeProjector.x, this->gridSizeProjector.y, this->gridSizeProjector.z);
  /// Init memory
  InitGpuMemory(inputSinogram, outputImage, typeOfProjector);
  
  initTime = clock();
  /// Pongo en cero la imagen de corrección, y hago la backprojection.
  checkCudaErrors(cudaMemset(d_image, 0,sizeof(float)*numPixels));
  switch(typeOfProjector)
  {
    case SIDDON_CYLINDRICAL_SCANNER:
      projector->Backproject(d_projection, d_image, d_ring1_mm, d_ring2_mm, (Sinogram3DCylindricalPet*)inputSinogram, outputImage, false);
      break;
  }
  finalTime = clock();
  printf(" Execution Time: %f sec\n", (float)(finalTime-initTime)/(float)CLOCKS_PER_SEC);
}

bool CuProjectorInterface::Project(Image* image, Sinogram3D* projection)
{
  clock_t initTime, finalTime;
  // Número total de píxeles.
  int numPixels = image->getPixelCount();
  // Número total de bins del sinograma:
  int numBins, numSinograms;
  numBins = projection->getBinCount();
  printf("######## PROJECTION #########");
  // INICALIZACIÓN DE GPU 
  if(!initCuda (gpuId))
  {
    return false;
  }
  printf(" Kernel Configuration: Block %d[x] x %d[y] x %d[z]. Grid %d[x] x %d[y] x %d[z]\n", this->blockSizeProjector.x, this->blockSizeProjector.y, this->blockSizeProjector.z,
    this->gridSizeProjector.x, this->gridSizeProjector.y, this->gridSizeProjector.z);
  /// Init memory
  InitGpuMemory(projection, image, typeOfProjector);
  
  initTime = clock();
  /// Pongo en cero la proyección estimada, y hago la backprojection.
  checkCudaErrors(cudaMemset(d_projection, 0,sizeof(float)*numBins));
  /// Proyección de la imagen:
  switch(typeOfProjector)
  {
    case SIDDON_CYLINDRICAL_SCANNER:
      projector->Project(d_image, d_projection, d_ring1_mm, d_ring2_mm, image, (Sinogram3DCylindricalPet*)projection, false);
      break;
  }
  finalTime = clock();
  printf(" Execution Time: %f sec\n", (float)(finalTime-initTime)/(float)CLOCKS_PER_SEC);
}


