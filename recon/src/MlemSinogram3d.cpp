
#include <MlemSinogram3d.h>


MlemSinogram3d::MlemSinogram3d(Sinogram3D* cInputProjection, Image* cInitialEstimate, string cPathSalida, string cOutputPrefix, int cNumIterations, int cSaveIterationInterval, bool cSaveIntermediate, bool cSensitivityImageFromFile, Projector* cForwardprojector, Projector* cBackprojector) : Mlem(cInitialEstimate, cPathSalida, cOutputPrefix, cNumIterations, cSaveIterationInterval, cSaveIntermediate, cSensitivityImageFromFile, cForwardprojector,  cBackprojector)
{ 
  inputProjection = cInputProjection->Copy();
}

MlemSinogram3d::MlemSinogram3d(string configFilename)
{
	/// Inicializo las variables con sus valores por default
	
}

/// Método que carga los coeficientes de corrección de atenuación desde un archivo interfile para aplicar como corrección.
bool MlemSinogram3d::setAcfProjection(string acfFilename)
{
  // Los leo como cylindrical, total para la corrección lo único que importa es realizar la resta bin a bin:
  attenuationCorrectionFactorsProjection = new Sinogram3DCylindricalPet((char*)acfFilename.c_str(), inputProjection->getRadioFov_mm(), inputProjection->getAxialFoV_mm(), inputProjection->getRadioFov_mm());
  enableAttenuationCorrection = true;  
}
    
/// Método que carga un sinograma desde un archivo interfile con la estimación de scatter para aplicar como corrección.
bool MlemSinogram3d::setScatterCorrectionProjection(string scatterFilename)
{
  // Los leo como cylindrical, total para la corrección lo único que importa es realizar la resta bin a bin:
  scatterCorrectionProjection = new Sinogram3DCylindricalPet((char*)scatterFilename.c_str(), inputProjection->getRadioFov_mm(), inputProjection->getAxialFoV_mm(), inputProjection->getRadioFov_mm());

  enableScatterCorrection = true;
}
    
/// Método que carga un sinograma desde un archivo interfile con la estimación de randomc para aplicar como corrección.
bool MlemSinogram3d::setRandomCorrectionProjection(string randomsFilename)
{
  // Los leo como cylindrical, total para la corrección lo único que importa es realizar la resta bin a bin:
  randomsCorrectionProjection = new Sinogram3DCylindricalPet((char*)randomsFilename.c_str(), inputProjection->getRadioFov_mm(), inputProjection->getAxialFoV_mm(), inputProjection->getRadioFov_mm());

  enableRandomsCorrection = true;
}

/// Método que aplica las correcciones habilitadas según se hayan cargado los sinogramas de atenuación, randoms y/o scatter.
bool MlemSinogram3d::correctInputSinogram()
{
  for(int i = 0; i < inputProjection->getNumSegments(); i++)
  {
    for(int j = 0; j < inputProjection->getSegment(i)->getNumSinograms(); j++)
    {
      for(int k = 0; k < inputProjection->getSegment(i)->getSinogram2D(j)->getNumProj(); k++)
      {
	for(int l = 0; l < inputProjection->getSegment(i)->getSinogram2D(j)->getNumR(); l++)
	{
	  if(enableScatterCorrection)
	  {
	    inputProjection->getSegment(i)->getSinogram2D(j)->setSinogramBin(k,l, inputProjection->getSegment(i)->getSinogram2D(j)->getSinogramBin(k,l)-
	      scatterCorrectionProjection->getSegment(i)->getSinogram2D(j)->getSinogramBin(k,l));
	  }
	  if(enableRandomsCorrection)
	  {
	    inputProjection->getSegment(i)->getSinogram2D(j)->setSinogramBin(k,l, inputProjection->getSegment(i)->getSinogram2D(j)->getSinogramBin(k,l)-
	      randomsCorrectionProjection->getSegment(i)->getSinogram2D(j)->getSinogramBin(k,l));
	  }
	  // Verifico que no quedo ningún bin negativo:
	  if(inputProjection->getSegment(i)->getSinogram2D(j)->getSinogramBin(k,l) < 0)
	  {
	    inputProjection->getSegment(i)->getSinogram2D(j)->setSinogramBin(k,l,0);
	  }
	  
	  // Por último, aplico la corrección por atenuación:
	  if(enableAttenuationCorrection)
	  {
	    inputProjection->getSegment(i)->getSinogram2D(j)->setSinogramBin(k,l,inputProjection->getSegment(i)->getSinogram2D(j)->getSinogramBin(k,l)*
	      attenuationCorrectionFactorsProjection->getSegment(i)->getSinogram2D(j)->getSinogramBin(k,l));
	  }
	}
      }
    }
  }
  /*char c_string[100];
  sprintf(c_string, "%s_correctedSinogram", this->outputFilenamePrefix.c_str()); 
  inputProjection->writeInterfile(c_string);*/
}
    
/// Método público que realiza la reconstrucción en base a los parámetros pasados al objeto Mlem instanciado
bool MlemSinogram3d::Reconstruct()
{
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
	  if(computeSensitivity(sensitivityImage)==false)
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
	  printf("Iteración Nº: %d\n", t);
	  /// Proyección de la imagen:
	  forwardprojector->Project(reconstructionImage, estimatedProjection);
	  clock_t finalClockProjection = clock();
	  /// Guardo el likelihood (Siempre va una iteración atrás, ya que el likelihhod se calcula a partir de la proyección
	  /// estimada, que es el primer paso del algoritmo). Se lo calculo al sinograma
	  /// proyectado, respecto del de entrada.
	  this->likelihoodValues[t] = estimatedProjection->getLikelihoodValue(inputProjection);
	  /// Pongo en cero la proyección estimada, y hago la backprojection.
	  backprojectedImage->fillConstant(0);
	  backprojector->DivideAndBackproject(inputProjection, estimatedProjection, backprojectedImage);
	  clock_t finalClockBackprojection = clock();
	  /// Actualización del Pixel
	  for(int k = 0; k < nPixels; k++)
	  {
	      /// Si el coeficiente de sensitivity es menor que 1 puedo, plantear distintas alternativas, pero algo
	      /// hay que hacer sino el valor del píxel tiende a crecer demasiado. .
	      if(ptrSensitivityPixels[k]>=updateThreshold)
	      {
		ptrPixels[k] = ptrPixels[k] * ptrBackprojectedPixels[k] / ptrSensitivityPixels[k];
	      }
	      else
	      {
		/// Si la sensitivity image es distinta de cero significa que estoy fuera del fov de reconstrucción
		/// por lo que pongo en cero dicho píxel:
		ptrPixels[k] = 0;
	      }
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
	      if(saveIntermediateProjectionAndBackprojectedImage)
	      {
		// Tengo que guardar la estimated projection, y la backprojected image.
		sprintf(c_string, "%s_projection_iter_%d", outputFilenamePrefix.c_str(), t); /// La extensión se le agrega en write interfile.
		outputFilename.assign(c_string);
		estimatedProjection->writeInterfile((char*)outputFilename.c_str());
		sprintf(c_string, "%s_backprojected_iter_%d", outputFilenamePrefix.c_str(), t); /// La extensión se le agrega en write interfile.
		outputFilename.assign(c_string);
		backprojectedImage->writeInterfile((char*)outputFilename.c_str());
	      }
	    }
	  }
	  clock_t finalClockIteration = clock();
	  /// Cargo los tiempos:
	  timesIteration_mseg[t] = (float)(finalClockIteration-initialClockIteration)*1000/(float)CLOCKS_PER_SEC;
	  timesBackprojection_mseg[t] = (float)(finalClockBackprojection-finalClockProjection)*1000/(float)CLOCKS_PER_SEC;
	  timesForwardprojection_mseg[t] = (float)(finalClockProjection-initialClockIteration)*1000/(float)CLOCKS_PER_SEC;
	  timesPixelUpdate_mseg[t] = (float)(finalClockIteration-finalClockBackprojection)*1000/(float)CLOCKS_PER_SEC;

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

bool MlemSinogram3d::computeSensitivity(Image* outputImage)
{
  /// Creo un Sinograma ·D igual que el de entrada.
  Sinogram3D* constantSinogram2D = inputProjection->Copy();
  /// Lo lleno con un valor constante
  constantSinogram2D->FillConstant(1);
  /// Por último hago la backprojection
  backprojector->Backproject(constantSinogram2D, outputImage);
  // Umbral para la actualización de píxel:
  updateUpdateThreshold();
  return true;
}