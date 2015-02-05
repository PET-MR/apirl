
#include <Mlem.h>


Mlem::Mlem(Image* cInitialEstimate, string cPathSalida, string cOutputPrefix, int cNumIterations, int cSaveIterationInterval, bool cSaveIntermediate, bool cSensitivityImageFromFile, Projector* cForwardprojector, Projector* cBackprojector)
{
  initialEstimate = cInitialEstimate;
  pathSalida.assign(cPathSalida);
  outputFilenamePrefix.assign(cOutputPrefix);
  numIterations = cNumIterations;
  saveIterationInterval = cSaveIterationInterval;
  saveIntermediateProjectionAndBackprojectedImage = cSaveIntermediate;
  sensitivityImageFromFile = cSensitivityImageFromFile;
  forwardprojector = cForwardprojector;
  backprojector = cBackprojector;
  // Las correciones por defecto deshabilitadas:
  enableAttenuationCorrection = false;
  enableRandomsCorrection = false;
  enableScatterCorrection = false;
  enableNormalization = false;
  
  sizeReconImage = initialEstimate->getSize();
  reconstructionImage = new Image(sizeReconImage);
  /// Inicializo el Sensitivity Image (Pero todavía no se calcula)
  /// Le paso la estructura con el tamaño de la misma a partir de la reconstruction image.
  sensitivityImage = new Image(reconstructionImage->getSize());
  /// Nombre del archivo de log:
  logFileName.assign(this->outputFilenamePrefix);
  logFileName.append(".log");
  /// Inicializo en NULL los likelihood values:
  likelihoodValues = NULL;
}

Mlem::Mlem(string configFilename)
{
	/// Inicializo las variables con sus valores por default
	
}

void Mlem::updateUpdateThreshold()
{
  updateThreshold = sensitivityImage->getMinValue() + (sensitivityImage->getMaxValue()-sensitivityImage->getMinValue()) * 0.001;
  
}


bool Mlem::setAttenuationImage(string attenImageFilename)
{
  attenuationImage = new Image(attenImageFilename);
  
  // Leo la imagen de atenuación desde el archivo, si hay algún error, lo cargo.
  /*if(!attenuationImage->readFromInterfile((char*) attenImageFilename.c_str()))
  {
	strError.assign("Error al leer el archivo interfile.");
	return false;
  }*/
  
  // Ahora verifico el tamaño de la imagen, si bien la cantidad de píxeles no debe ser
  // la misma obligatoriamente, el field of view si debería serlo.
  SizeImage sizeAttenuationImage = attenuationImage->getSize();
  if(attenuationImage->getFovRadio() != initialEstimate->getFovRadio())
  {
	strError.assign("El radio del FoV de la imagen de atenuación es distinto al del initial estimate.");
	return false;
  }
  if(attenuationImage->getFovHeight() != initialEstimate->getFovHeight())
  {
	strError.assign("La altura (z) del FoV de la imagen de atenuación es distinto al del initial estimate.");
	return false;
  }
  if((attenuationImage->getSize().nPixelsX != initialEstimate->getSize().nPixelsX) || 
	(attenuationImage->getSize().nPixelsY != initialEstimate->getSize().nPixelsY) ||
	(attenuationImage->getSize().nPixelsZ != initialEstimate->getSize().nPixelsZ))
  {
	cout<<"Warning: La imagen de atenuación es de distinto tamaño que el inital estimate. Se continua procesando"<<
	  " ya que los FoV son iguales." << endl;
  }
  
  // Ya tengo un mapa de atenuación que cumple con los requerimientos, ahora debo calcular la proyección
  // con los factores de corrección. Eso lo hago en cada Mlem con una sobrecarga.
  enableAttenuationCorrection = true;
  return true;
}


