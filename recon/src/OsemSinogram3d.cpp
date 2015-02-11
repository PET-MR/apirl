#include <OsemSinogram3d.h>


OsemSinogram3d::OsemSinogram3d(Sinogram3D* cInputProjection, Image* cInitialEstimate, string cPathSalida, string cOutputPrefix, int cNumIterations, int cSaveIterationInterval, bool cSaveIntermediate, bool cSensitivityImageFromFile, Projector* cForwardprojector, Projector* cBackprojector, int cNumSubsets) : MlemSinogram3d(cInputProjection, cInitialEstimate, cPathSalida, cOutputPrefix, cNumIterations, cSaveIterationInterval, cSaveIntermediate, cSensitivityImageFromFile, cForwardprojector,  cBackprojector)
{
  numSubsets = cNumSubsets;
  // Tengo que crear la sensitivity images y el vector de thresholds:
  updateThresholds = (float*) malloc(sizeof(float)*numSubsets);
  sensitivityImages = (Image**) malloc(sizeof(Image*)*numSubsets);
  // Ahora incializo cada imagen, debe ser del mismo tamaño que la de salida, y todas en cero:
  for(int s=0; s < numSubsets; s++)
  {
    sensitivityImages[s] = new Image(cInitialEstimate);
    sensitivityImages[s]->fillConstant(0);
  }
}

OsemSinogram3d::OsemSinogram3d(string configFilename):MlemSinogram3d(configFilename)
{
    /// Inicializo las variables con sus valores por default	
}

/// Override the updateUpdateThreshold function of mlem. For osem we have an update threshold per subset.
void OsemSinogram3d::updateUpdateThreshold()
{
  for(int s = 0; s < numSubsets; s++)
  {
    updateThresholds[s] = sensitivityImages[s]->getMinValue() + (sensitivityImages[s]->getMaxValue()-sensitivityImages[s]->getMinValue()) * 0.0001;  
    #ifdef __DEBUG__
      printf("Threshold: %f\n", updateThresholds[s]);
    #endif
  }
}

/// Método público que realiza la reconstrucción en base a los parámetros pasados al objeto Mlem instanciado
bool OsemSinogram3d::Reconstruct()
{
  /// Tamaño de la imagen:
  SizeImage sizeImage = reconstructionImage->getSize();
  /// Proyección auxiliar, donde guardo el sinograma proyectado:
  Sinogram3D* estimatedProjection; // = new Sinogram3D(inputProjection);
  Sinogram3D *inputSubset;
  Sinogram3D *normalizationSubset; // Pointer to a sinogram3d to get each subset of normalization factors.
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
  char c_string[512]; string outputFilename;
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
    /// Leo todas las imágenes:
    for(int s = 0; s < numSubsets; s++)
    {
      /// Leo las distintas imágnes de sensibilidad. Para osem el sensitivity filename, tiene que tener el
      /// prefijo de los nombres, y luego de les agrega un _%d, siendo %d el índice de subset.
      sprintf(c_string, "%s_subset_%d.h33", sensitivityFilename.c_str(), s);
      /// Leo el sensitivity volume desde el archivo
      sensitivityImages[s]->readFromInterfile(c_string);
      sensitivityImages[s]->forcePositive();
    }
  }
  else
  {
    /// Calculo todas los sensitivty volume, tengo tantos como subset. Sino alcancanzar la ram
    /// para almacenar todos, debería calcularlos dentro del for por cada iteración:
    for(int s = 10; s < numSubsets; s++)
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
      // Sometimes it can be a negative number in a border (precision problems):
      sensitivityImages[s]->forcePositive();
    }
  }
  // Update the thresholds:
  updateUpdateThreshold();
  
  /// Inicializo el volumen a reconstruir con la imagen del initial estimate:
  reconstructionImage = new Image(initialEstimate);
  ptrPixels = reconstructionImage->getPixelsPtr();
  /// Hago el log de la reconstrucción:
  Logger* logger = new Logger(logFileName);
  /// Escribo el título y luego los distintos parámetros de la reconstrucción:
  logger->writeLine("######## ML-EM Reconstruction #########");
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
      /// Si hay normalización, la aplico luego de la proyección:
      if(enableNormalization)
      {
	normalizationSubset = normalizationCorrectionFactorsProjection->getSubset(s, numSubsets);
	estimatedProjection->multiplyBinToBin(normalizationSubset);
      }
      // Si hay que guardar la proyección, lo hago acá porque después se modifica:
      if((saveIterationInterval != 0) && ((t%saveIterationInterval)==0) && saveIntermediateProjectionAndBackprojectedImage)
      {
	// Tengo que guardar la estimated projection, y la backprojected image.
	sprintf(c_string, "%s_projection_iter_%d", outputFilenamePrefix.c_str(), t); /// La extensión se le agrega en write interfile.
	outputFilename.assign(c_string);
	estimatedProjection->writeInterfile((char*)outputFilename.c_str());
      }
      //clock_t finalClockProjection = clock();
      /// Guardo el likelihood (Siempre va una iteración atrás, ya que el likelihhod se calcula a partir de la proyección
      /// estimada, que es el primer paso del algoritmo). Se lo calculo al sinograma
      /// proyectado, respecto del de entrada.
      this->likelihoodValues[t] = estimatedProjection->getLikelihoodValue(inputSubset);
      /// Pongo en cero la proyección estimada, y hago la backprojection.
      backprojectedImage->fillConstant(0);
      //backprojector->DivideAndBackproject(inputSubset, estimatedProjection, backprojectedImage);
      /// Divido input sinogram por el estimated:
      estimatedProjection->inverseDivideBinToBin(inputSubset);
      /// Si hay normalización, la aplico luego de la proyección:
      if(enableNormalization)
      {
	// The subset was already generated in the projection process.
	estimatedProjection->multiplyBinToBin(normalizationSubset);
	// Free memory
	delete normalizationSubset;
      }
      /// Retroproyecto
      backprojector->Backproject(estimatedProjection, backprojectedImage);
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
      if((saveIterationInterval != 0) && ((t%saveIterationInterval)==0))
      {
	// Cuando me piden guardar a cada intervalo solo guardo una imagen del subset al menos que esta habilitado el
	// saveIntermediateProjectionAndBackprojectedImage
	if(saveIntermediateProjectionAndBackprojectedImage)
	{
	  sprintf(c_string, "%s_iter_%d_subset_%d", outputFilenamePrefix.c_str(), t,s); /// La extensión se le agrega en write interfile.
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
  Sinogram3D* backprojectSinogram3D;
  char c_string[100];
  /// Si no hay normalización lo lleno con un valor constante, de lo contrario bakcprojec normalizacion:
  if (enableNormalization)
    backprojectSinogram3D = normalizationCorrectionFactorsProjection->getSubset(indexSubset, numSubsets);
  else
  {
    backprojectSinogram3D = inputProjection->getSubset(indexSubset, numSubsets);
    backprojectSinogram3D->FillConstant(1);
  }
  // Tengo que guardar la estimated projection, y la backprojected image.
	sprintf(c_string, "%s_projection_sens_%d_", outputFilenamePrefix.c_str(), indexSubset); /// La extensión se le agrega en write interfile.
	backprojectSinogram3D->writeInterfile((char*)c_string);
  /// Por último hago la backprojection
  backprojector->Backproject(backprojectSinogram3D, outputImage);
  /// Free memory:
  delete backprojectSinogram3D;
  return true;
}