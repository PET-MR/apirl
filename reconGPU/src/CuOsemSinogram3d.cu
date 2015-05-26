#include <CuOsemSinogram3d.h>
#include "../kernels/CuSiddonProjector_kernels.cu"

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

__device__ __constant__ int d_numPixelsPerSlice;

__device__ __constant__ int d_numBinsSino2d;

extern texture<float, 3, cudaReadModeElementType> texImage;  // 3D texture

extern surface<void, 3> surfImage;
  
#endif

CuOsemSinogram3d::CuOsemSinogram3d(Sinogram3D* cInputProjection, Image* cInitialEstimate, string cPathSalida, string cOutputPrefix, int cNumIterations, int cSaveIterationInterval, bool cSaveIntermediate, bool cSensitivityImageFromFile, Projector* cForwardprojector, Projector* cBackprojector, int cNumSubsets) : OsemSinogram3d(cInputProjection, cInitialEstimate, cPathSalida, cOutputPrefix, cNumIterations, cSaveIterationInterval, cSaveIntermediate, cSensitivityImageFromFile, cForwardprojector, cBackprojector, cNumSubsets)
{

}

CuOsemSinogram3d::CuOsemSinogram3d(string configFilename):OsemSinogram3d(configFilename)
{
    /// Inicializo las variables con sus valores por default
    
}

bool CuOsemSinogram3d::initCuda (int device, Logger* logger)
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

/// Método que configura los tamaños de ejecución del kernel de proyección.
void CuOsemSinogram3d::setProjectorKernelConfig(unsigned int numThreadsPerBlockX, unsigned int numThreadsPerBlockY, unsigned int numThreadsPerBlockZ, 
			      unsigned int numBlocksX, unsigned int numBlocksY, unsigned int numBlocksZ)
{
  blockSizeProjector = dim3(numThreadsPerBlockX, numThreadsPerBlockY, numThreadsPerBlockZ);
  gridSizeProjector = dim3(numBlocksX, numBlocksY, numBlocksZ);
}

/// Método que configura los tamaños de ejecución del kernel de retroproyección.
void CuOsemSinogram3d::setBackprojectorKernelConfig(unsigned int numThreadsPerBlockX, unsigned int numThreadsPerBlockY, unsigned int numThreadsPerBlockZ, 
				  unsigned int numBlocksX, unsigned int numBlocksY, unsigned int numBlocksZ)
{
  blockSizeBackprojector = dim3(numThreadsPerBlockX, numThreadsPerBlockY, numThreadsPerBlockZ);
  gridSizeBackprojector = dim3(numBlocksX, numBlocksY, numBlocksZ);
}

/// Método que configura los tamaños de ejecución del kernel de actualización de píxel.
void CuOsemSinogram3d::setUpdatePixelKernelConfig(unsigned int numThreadsPerBlockX, unsigned int numThreadsPerBlockY, unsigned int numThreadsPerBlockZ, 
				unsigned int numBlocksX, unsigned int numBlocksY, unsigned int numBlocksZ)
{
  blockSizeImageUpdate = dim3(numThreadsPerBlockX, numThreadsPerBlockY, numThreadsPerBlockZ);
  gridSizeImageUpdate = dim3(numBlocksX, numBlocksY, numBlocksZ);  
}

bool CuOsemSinogram3d::InitGpuMemory()
{
  // Número total de píxeles.
  int numPixels = reconstructionImage->getPixelCount();
  // Número total de bins del sinograma:
  int numBins = inputProjection->getBinCount();
  // Número total de bins por subset, que es el que voy a tener que usar para los proyectores:
  int numBinsSubset = inputProjection->getSubset(0)->getBinCount();
  // Lo mismo para el numero de sinogramas:
  int numSinograms = inputProjection->getNumSinograms();
  // Pido memoria para la gpu, debo almacenar los sinogramas y las imágenes.
  // Lo hago acá y no en el proyector para mantenerme en memmoria de gpu durante toda la reconstrucción.
  
  checkCudaErrors(cudaMalloc((void**) &d_reconstructionImage, sizeof(float)*numPixels));
  checkCudaErrors(cudaMalloc((void**) &d_backprojectedImage, sizeof(float)*numPixels));
  // Para la proyección estimada siempre va a ser del tamaño del subset.
  checkCudaErrors(cudaMalloc((void**) &d_estimatedProjection, sizeof(float)*numBinsSubset));
  checkCudaErrors(cudaMalloc((void**) &d_ring1, sizeof(float)*inputProjection->getNumSinograms()));
  checkCudaErrors(cudaMalloc((void**) &d_ring2, sizeof(float)*inputProjection->getNumSinograms()));
  // Por ahora tengo las dos, d_ring1 me da el índice de anillo, y d_ring1_mm me da directamente la coordenada axial.
  // Agregue esto porque para usar una única LOR para
  checkCudaErrors(cudaMalloc((void**) &d_ring1_mm, sizeof(int)*numSinograms));
  checkCudaErrors(cudaMalloc((void**) &d_ring2_mm, sizeof(int)*numSinograms));
  // Para la sensitivity iamge, tengo un array de sensitivity images:
  checkCudaErrors(cudaMalloc((void***) &d_sensitivityImages, sizeof(float*)*numSubsets));	// Memory for the array.
  // Para los subsets lo mismo:
  checkCudaErrors(cudaMalloc((void***) &d_inputProjectionSubsets, sizeof(float*)*numSubsets));
  for(int i = 0; i < numSubsets; i++)
  {
    // Memoria para cada sensitivty image:
    checkCudaErrors(cudaMalloc((void**) &(d_sensitivityImages[i]), sizeof(float)*numPixels));	// Memoria para la imagen.
    // Pongo en cero la imágens de sensibilidad:
    checkCudaErrors(cudaMemset(d_sensitivityImages[i], 0,sizeof(float)*numPixels));
    // Memoria para cada subset del sinograma de entrada:
    checkCudaErrors(cudaMalloc((void**) &(d_inputProjectionSubsets[i]), sizeof(float)*numBinsSubset));
    // Copio el subset del sinograma de entrada, llamo a una función porque tengo que ir recorriendo todos los sinogramas:
    
    CopySinogram3dHostToGpu(d_inputProjectionSubsets[i], inputProjection->getSubset(i));
  }
  // Copio la iamgen inicial:
  checkCudaErrors(cudaMemcpy(d_reconstructionImage, initialEstimate->getPixelsPtr(),sizeof(float)*numPixels,cudaMemcpyHostToDevice));
  // Pongo en cero la imágen de retroproyección:
  checkCudaErrors(cudaMemset(d_backprojectedImage, 0,sizeof(float)*numPixels));
  // Copio el sinograma de entrada, llamo a una función porque tengo que ir recorriendo todos los sinogramas:
  CopySinogram3dHostToGpu(d_inputProjection, inputProjection); // No se si realmente lo necesito.
  // Pongo en cero el sinograma de proyección:
  checkCudaErrors(cudaMemset(d_estimatedProjection, 0,sizeof(float)*numBinsSubset));
  
  // Además de copiar los valores de todos los bins, debo inicializar todas las constantes de reconstrucción.
  // Por un lado tengo los valores de coordenadas posibles de r, theta y z. Los mismos se copian a memoria constante de GPU (ver vectores globales al inicio de este archivo.
  // Los theta values los tengo que cargar por cada subset.
  // checkCudaErrors(cudaMemcpyToSymbol(d_thetaValues_deg, inputProjection->getAngPtr(), sizeof(float)*inputProjection->getNumProj()));
  checkCudaErrors(cudaMemcpyToSymbol(d_RValues_mm, inputProjection->getRPtr(), sizeof(float)*inputProjection->getNumR()));
  checkCudaErrors(cudaMemcpyToSymbol(d_AxialValues_mm, inputProjection->getAxialPtr(), sizeof(float)*inputProjection->getNumRings()));
  SizeImage size =  reconstructionImage->getSize();
  checkCudaErrors(cudaMemcpyToSymbol(d_imageSize, &size, sizeof(reconstructionImage->getSize())));
  float aux;
  aux = inputProjection->getRadioFov_mm();
  checkCudaErrors(cudaMemcpyToSymbol(d_RadioFov_mm, &aux, sizeof(inputProjection->getRadioFov_mm())));
  aux = inputProjection->getAxialFoV_mm();
  checkCudaErrors(cudaMemcpyToSymbol(d_AxialFov_mm, &aux, sizeof(inputProjection->getAxialFoV_mm())));

  // Para el sinograma 3d tengo que cada sino 2d puede representar varios sinogramas asociados a distintas combinaciones de anillos.
  // En la versión con CPU proceso todas las LORs, ahora solo voy a considerar la del medio, que sería la ventaja de reducir el volumen de LORs.
  // Entonces genero un array con las coordenadas de anillos de cada sinograma.
  int iSino = 0;
  float* auxRings1 = new float[inputProjection->getNumSinograms()];
  float* auxRings2 = new float[inputProjection->getNumSinograms()];
  float* auxRings1_mm = (float*)malloc(sizeof(float)*numSinograms);
  float* auxRings2_mm = (float*)malloc(sizeof(float)*numSinograms);
  float numZ;
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
  checkCudaErrors(cudaMemcpy(d_ring1, auxRings1, sizeof(float)*inputProjection->getNumSinograms(), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_ring2, auxRings2, sizeof(float)*inputProjection->getNumSinograms(), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_ring1_mm, auxRings1_mm, sizeof(float)*numSinograms, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_ring2_mm, auxRings2_mm, sizeof(float)*numSinograms, cudaMemcpyHostToDevice));

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
  texImage.addressMode[0] = cudaAddressModeClamp;   // wrap texture coordinates
  texImage.addressMode[1] = cudaAddressModeClamp;
  texImage.addressMode[2] = cudaAddressModeClamp;
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

bool CuOsemSinogram3d::InitSubsetConstants(int indexSubset)
{
  checkCudaErrors(cudaMemcpyToSymbol(d_thetaValues_deg, inputProjection->getSubset(indexSubset, numSubsets)->getAngPtr(), sizeof(float)*inputProjection->getSubset(indexSubset, numSubsets)->getNumProj()));
}

int CuOsemSinogram3d::CopySinogram3dHostToGpu(float* d_destino, Sinogram3D* h_source)
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

// Método de reconstrucción que no se le indica el índice de GPU, incializa la GPU 0 por defecto.
bool CuOsemSinogram3d::Reconstruct(TipoProyector tipoProy)
{
  Reconstruct(0);
}

/// Método público que realiza la reconstrucción en base a los parámetros pasados al objeto Mlem instanciado
bool CuOsemSinogram3d::Reconstruct(TipoProyector tipoProy, int indexGpu)
{
  /// Hago el log de la reconstrucción:
  Logger* logger = new Logger(logFileName);
  // INICALIZACIÓN DE GPU 
  if(!initCuda (0, logger))
  {
	  return false;
  }
  /// Tamaño de la imagen:
  SizeImage sizeImage = reconstructionImage->getSize();
  /// Proyección auxiliar, donde guardo el sinograma proyectado:
  Sinogram3D* estimatedProjection; // = new Sinogram3D(inputProjection);
  Sinogram3D *inputSubset;
  //estimatedProjection->FillConstant(0);
  /// Imagen donde guardo la backprojection.
  Image* backprojectedImage = new Image(reconstructionImage->getSize());
  /// Puntero a la imagen.
  float* ptrPixels = reconstructionImage->getPixelsPtr();
  /// Puntero a la sensitivity image.
  float* ptrSensitivityPixels;
  /// Puntero a la sensitivity image.
  float* ptrBackprojectedPixels = backprojectedImage->getPixelsPtr();
  /// Puntero del array con los tiempos de reconstrucción por iteración.
  float* timesIteration_mseg;
  float timesTotalIteration_seg;
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
  /// Calculo todas los sensitivty volume, tengo tantos como subset. Sino alcancanzar la ram
  /// para almacenar todos, debería calcularlos dentro del for por cada iteración:
  for(int s = 0; s < numSubsets; s++)
  {
    /// Calculo el sensitivity volume
    if(computeSensitivity(sensitivityImages[s], s)==false)
    {
      strError = "Error al calcular la sensitivity Image.";
      return false;
    }
    // La guardo en disco.
    string sensitivityFileName = outputFilenamePrefix;
    sprintf(c_string, "_sensitivity_subset_%d", s);
    sensitivityFileName.append(c_string);
    sensitivityImages[s]->writeInterfile((char*)sensitivityFileName.c_str());
    updateThresholds[s] = sensitivityImages[s]->getMaxValue()*0.05;
  }
  /// Inicializo el volumen a reconstruir con la imagen del initial estimate:
  reconstructionImage = new Image(initialEstimate);
  ptrPixels = reconstructionImage->getPixelsPtr();
  
  /// Escribo el título y luego los distintos parámetros de la reconstrucción:
  logger->writeLine("######## CUDA OS-EM Reconstruction #########");
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

  // También se realiza un registro de los tiempos de ejecución:
  clock_t initialClock = clock();
  //Start with the iteration
  printf("Iniciando Reconstrucción...\n");
  /// Arranco con el log de los resultados:
  strcpy(c_string, "_______RECONSTRUCCION_______");
  logger->writeLine(c_string, strlen(c_string));
  /// Voy generando mensajes con los archivos creados en el log de salida:
  int nPixels = reconstructionImage->getPixelCount();
  int nBins = inputProjection->getSubset(0)->getBinCount();	// The number of bins that we use in the projection and backrpojection is the one of the subset.
  for(unsigned int t = 0; t < this->numIterations; t++)
  {
    printf("Iteration Nº: %d\n", t+1);
    timesTotalIteration_seg = 0;
    // Por cada iteración debo repetir la operación para todos los subsets.
    for(unsigned int s = 0; s < this->numSubsets; s++)
    {
      printf("\tsubiteration Nº: %d", s+1);
      clock_t initialClockIteration = clock();
      // Init constants for the subset.
      InitSubsetConstants(s);
      // Now I need to use the sinogram d_inputProjectionSubsets[s].
      /// Pongo en cero la proyección estimada, y hago la backprojection.
      checkCudaErrors(cudaMemset(d_estimatedProjection, 0,sizeof(float)*nBins));
      /// Proyección de la imagen:
      switch(tipoProy)
      {
	case SIDDON_PROJ_TEXT_CYLINDRICAL_SCANNER: // This siddon implementation has only projection, so it uses the standard backprojection.
	  CopyDevImageToTexture(d_reconstructionImage, reconstructionImage->getSize()); // Copy the reconstruction iamge to texture.
	  forwardprojector->Project(d_reconstructionImage, d_estimatedProjection, d_ring1_mm, d_ring2_mm, reconstructionImage, (Sinogram3DCylindricalPet*)inputProjection->getSubset(s, numSubsets), false); // Debo pasa el subset para tener el tamaño correcto de sino.
	  break;
	case SIDDON_CYLINDRICAL_SCANNER:
	  forwardprojector->Project(d_reconstructionImage, d_estimatedProjection, d_ring1_mm, d_ring2_mm, reconstructionImage, (Sinogram3DCylindricalPet*)inputProjection->getSubset(s, numSubsets), false);
	  break;
      }
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
      
      /// Pongo en cero la proyección estimada, y hago la backprojection.
      backprojectedImage->fillConstant(0);
      backprojector->DivideAndBackproject(inputSubset, estimatedProjection, backprojectedImage);
      //clock_t finalClockBackprojection = clock();
      
      // Obtengo el puntero de la snesitivity image (Si no me alcanzar la ram para tenrlas todas almacenadas, acá debería calcularla):
      ptrSensitivityPixels = sensitivityImages[s]->getPixelsPtr();
      /// Actualización del Pixel
      for(int k = 0; k < nPixels; k++)
      {
	/// Si el coeficiente de sensitivity es menor que 1 puedo, plantear distintas alternativas, pero algo
	/// hay que hacer sino el valor del píxel tiende a crecer demasiado. .
	if(ptrSensitivityPixels[k]>=updateThresholds[s])
	//if(ptrSensitivityPixels[k]>=0)
	{
	  ptrPixels[k] = ptrPixels[k] * ptrBackprojectedPixels[k] / ptrSensitivityPixels[k];
	}
// 	else if(ptrSensitivityPixels[k]!=0)
// 	{
// 	  /// Lo mantengo igual, porque sino crece a infinito.
// 	  ptrPixels[k] = ptrPixels[k];
// 	}
	else
	{
	  /// Si la sensitivity image es distinta de cero significa que estoy fuera del fov de reconstrucción
	  /// por lo que pongo en cero dicho píxel:
	  ptrPixels[k] = 0;
	}
      }
      if(saveIterationInterval != 0)
      {
	if((t%saveIterationInterval)==0)
	{
	  // Cuando me piden guardar a cada intervalo solo guardo una imagen del subset al menos que esta habilitado el
	  // saveIntermediateProjectionAndBackprojectedImage
	  if(saveIntermediateProjectionAndBackprojectedImage)
	  {
	    sprintf(c_string, "%s_iter_%d_subset_%d", outputFilenamePrefix.c_str(), t,s); /// La extensión se le agrega en write interfile.
	    string outputFilename;
	    outputFilename.assign(c_string);
	    reconstructionImage->writeInterfile((char*)outputFilename.c_str());
	    // Tengo que guardar la estimated projection, y la backprojected image.
	    sprintf(c_string, "%s_projection_iter_%d_subset_%d", outputFilenamePrefix.c_str(), t); /// La extensión se le agrega en write interfile.
	    outputFilename.assign(c_string);
	    estimatedProjection->writeInterfile((char*)outputFilename.c_str());
	    sprintf(c_string, "%s_backprojected_iter_%d_subset_%d", outputFilenamePrefix.c_str(), t); /// La extensión se le agrega en write interfile.
	    outputFilename.assign(c_string);
	    backprojectedImage->writeInterfile((char*)outputFilename.c_str());
	  }
	}
      }
      // Elimino el subset.
      delete inputSubset;
      delete estimatedProjection;
    }
    /// Verifico
    if(saveIterationInterval != 0)
    {
      if((t%saveIterationInterval)==0)
      {
	sprintf(c_string, "%s_iter_%d", outputFilenamePrefix.c_str(), t); /// La extensión se le agrega en write interfile.
	string outputFilename;
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
    //timesBackprojection_mseg[t] = (float)(finalClockBackprojection-finalClockProjection)*1000/(float)CLOCKS_PER_SEC;
    //timesForwardprojection_mseg[t] = (float)(finalClockProjection-initialClockIteration)*1000/(float)CLOCKS_PER_SEC;
    //timesPixelUpdate_mseg[t] = (float)(finalClockIteration-finalClockBackprojection)*1000/(float)CLOCKS_PER_SEC;
  }

  clock_t finalClock = clock();
  sprintf(c_string, "%s_final", outputFilenamePrefix.c_str()); /// La extensión se le agrega en write interfile.
  string outputFilename;
  outputFilename.assign(c_string);
  reconstructionImage->writeInterfile((char*)outputFilename.c_str());
  /// Termino con el log de los resultados:
  sprintf(c_string, "Imagen final guardada en: %s", outputFilename.c_str());
  logger->writeLine(c_string);
  /// Calculo la proyección de la última imagen para poder calcular el likelihood final:
  forwardprojector->Project(reconstructionImage, estimatedProjection);
  this->likelihoodValues[this->numIterations] = estimatedProjection->getLikelihoodValue(inputProjection);

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

bool OsemSinogram3d::computeSensitivity(Image* outputImage, int indexSubset)
{
  /// Creo un Sinograma del subset correspondiente:
  Sinogram3D* constantSinogram3D = inputProjection->getSubset(indexSubset, numSubsets);
  /// Lo lleno con un valor constante
  constantSinogram3D->FillConstant(1);
  /// Por último hago la backprojection
  backprojector->Backproject(constantSinogram3D, outputImage);
  return true;
}