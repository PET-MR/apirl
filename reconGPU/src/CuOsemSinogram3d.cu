#include <CuOsemSinogram3d.h>

// Memoria constante con los valores de los angulos de la proyeccion,
__device__ __constant__ float d_thetaValues_deg[MAX_PHI_VALUES];

// Memoria constante con los valores de la distancia r.
__device__ __constant__ float d_RValues_mm[MAX_R_VALUES];

// Memoria constante con los valores de la coordenada axial o z.
__device__ __constant__ float d_AxialValues_mm[MAX_Z_VALUES];

__device__ __constant__ float d_RadioFov_mm;

__device__ __constant__ float d_AxialFov_mm;

__device__ __constant__ float d_RadioScanner_mm;

__device__ __constant__ SizeImage d_imageSize;

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
  // Pido memoria para la gpu, debo almacenar los sinogramas y las imágenes.
  // Lo hago acá y no en el proyector para mantenerme en memmoria de gpu durante toda la reconstrucción.
  checkCudaErrors(cudaMalloc((void**) &d_sensitivityImage, sizeof(float)*numPixels));
  checkCudaErrors(cudaMalloc((void**) &d_reconstructionImage, sizeof(float)*numPixels));
  checkCudaErrors(cudaMalloc((void**) &d_backprojectedImage, sizeof(float)*numPixels));
  checkCudaErrors(cudaMalloc((void**) &d_inputProjection, sizeof(float)*numBins));
  checkCudaErrors(cudaMalloc((void**) &d_estimatedProjection, sizeof(float)*numBins));
  checkCudaErrors(cudaMalloc((void**) &d_ring1, sizeof(float)*inputProjection->getNumSinograms()));
  checkCudaErrors(cudaMalloc((void**) &d_ring2, sizeof(float)*inputProjection->getNumSinograms()));
  // Copio la iamgen inicial:
  checkCudaErrors(cudaMemcpy(d_reconstructionImage, initialEstimate->getPixelsPtr(),sizeof(float)*numPixels,cudaMemcpyHostToDevice));
  // Pongo en cero la imágens de sensibilidad y la de retroproyección:
  checkCudaErrors(cudaMemset(d_sensitivityImage, 0,sizeof(float)*numPixels));
  checkCudaErrors(cudaMemset(d_backprojectedImage, 0,sizeof(float)*numPixels));
  // Copio el sinograma de entrada, llamo a una función porque tengo que ir recorriendo todos los sinogramas:
  CopySinogram3dHostToGpu(d_inputProjection, inputProjection);
  // Pongo en cero el sinograma de proyección:
  checkCudaErrors(cudaMemset(d_estimatedProjection, 0,sizeof(float)*numBins));
  
  // Además de copiar los valores de todos los bins, debo inicializar todas las constantes de reconstrucción.
  // Por un lado tengo los valores de coordenadas posibles de r, theta y z. Los mismos se copian a memoria constante de GPU (ver vectores globales al inicio de este archivo.
  checkCudaErrors(cudaMemcpyToSymbol(d_thetaValues_deg, inputProjection->getAngPtr(), sizeof(float)*inputProjection->getNumProj()));
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
  for(int i = 0; i < inputProjection->getNumSegments(); i++)
  {
    for(int j = 0; j < inputProjection->getSegment(i)->getNumSinograms(); j++)
    {
      // Como voy a agarrar solo la combinación del medio:
      int indexMedio = floor(inputProjection->getSegment(i)->getSinogram2D(j)->getNumZ()/2);
      auxRings1[iSino] = inputProjection->getSegment(i)->getSinogram2D(j)->getRing1FromList(indexMedio);
      auxRings2[iSino] = inputProjection->getSegment(i)->getSinogram2D(j)->getRing2FromList(indexMedio);
      iSino++;
    }
  }
  // Copio los índices de anillos a memoris de GPU:
  checkCudaErrors(cudaMemcpy(d_ring1, auxRings1, sizeof(float)*inputProjection->getNumSinograms(), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_ring2, auxRings2, sizeof(float)*inputProjection->getNumSinograms(), cudaMemcpyHostToDevice));

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
bool CuOsemSinogram3d::Reconstruct()
{
  Reconstruct(0);
}

/// Método público que realiza la reconstrucción en base a los parámetros pasados al objeto Mlem instanciado
bool CuOsemSinogram3d::Reconstruct(int indexGpu)
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
  for(unsigned int t = 0; t < this->numIterations; t++)
  {
    clock_t initialClockIteration = clock();
    // Por cada iteración debo repetir la operación para todos los subsets.
    for(unsigned int s = 0; s < this->numSubsets; s++)
    {
      // Tengo que generar el subset del sinograma correspondiente para reconstruir con ese:
      inputSubset = inputProjection->getSubset(s, numSubsets);
      // La estimated también la cambio, porque van cambiando los ángulos de la proyección.
      estimatedProjection = inputProjection->getSubset(s, numSubsets); 
      estimatedProjection->FillConstant(0);
      printf("Iteración Nº: %d\n", t);
      /// Proyección de la imagen:
      forwardprojector->Project(reconstructionImage, estimatedProjection);

      //clock_t finalClockProjection = clock();
      /// Guardo el likelihood (Siempre va una iteración atrás, ya que el likelihhod se calcula a partir de la proyección
      /// estimada, que es el primer paso del algoritmo). Se lo calculo al sinograma
      /// proyectado, respecto del de entrada.
      this->likelihoodValues[t] = estimatedProjection->getLikelihoodValue(inputSubset);
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