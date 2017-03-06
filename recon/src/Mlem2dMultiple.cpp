
#include <Mlem2dMultiple.h>
#include <algorithm>

Mlem2dMultiple::Mlem2dMultiple(Sinograms2DmultiSlice* cInputProjection, Image* cInitialEstimate, string cPathSalida, string cOutputPrefix, int cNumIterations, int cSaveIterationInterval, bool cSaveIntermediate, bool cSensitivityImageFromFile, Projector* cForwardprojector, Projector* cBackprojector) : Mlem(cInitialEstimate, cPathSalida, cOutputPrefix, cNumIterations, cSaveIterationInterval, cSaveIntermediate, cSensitivityImageFromFile, cForwardprojector,  cBackprojector)
{
  inputProjection = cInputProjection->Copy();
  //inputSinograms = cInputProjection;
  
  // Inicializo puntero con 1 byte para luego hacer el realloc cuando sea necesario:
  this->likelihoodValues = (float*)malloc(sizeof(float)*1);
  
  // Debo verificar que la cantidad de sinogramas es la misma que de slices de la imagen ya
  // que se genera un slice por cada sinograma. Además debo verficiar que el ancho del slice coincida
  // con la separación axial de los sinogramas.
  if(inputProjection->getNumSinograms() != this->initialEstimate->getSize().nPixelsZ)
  {
    cout << "Error en reconstrucción de múltiples sinogramas 2d: la cantidad de sinogramas 2d es distinta a la de slices del initial estimate." << endl;
  }
  //inputProjection->setRadioFov_mm(280);
}

Mlem2dMultiple::Mlem2dMultiple(string configFilename)
{
  /// Inicializo las variables con sus valores por default
  // Inicializo puntero con 1 byte para luego hacer el realloc cuando sea necesario:
  this->likelihoodValues = (float*)malloc(sizeof(float)*1);
}

bool Mlem2dMultiple::setMultiplicativeProjection(string multiplicativeFilename)
{
  // Problema porque sinogram2d no tiene get lor para hacer backprojection
  // Instead of create a new sinogram instantiating a new object, I copy from inputProject. This way, is independent of the type
  // of derived class od sinogram3d it is being used.
  multiplicativeProjection = inputProjection->Copy();
  multiplicativeProjection->readFromInterfile((char*)multiplicativeFilename.c_str());
  enableMultiplicativeTerm = true;
  return enableMultiplicativeTerm;
}

/// Método que carga un sinograma desde un archivo interfile con la estimación de scatter para aplicar como corrección.
bool Mlem2dMultiple::setAdditiveProjection(string additiveFilename)
{
  // Instead of create a new sinogram instantiating a new object, I copy from inputProject. This way, is independent of the type
  // of derived class od sinogram3d it is being used.
  additiveProjection = inputProjection->Copy();
  additiveProjection->readFromInterfile((char*)additiveFilename.c_str());
  enableAdditiveTerm = true;
  return enableAdditiveTerm;
}   

/// Método público que realiza la reconstrucción en base a los parámetros pasados al objeto Mlem instanciado
bool Mlem2dMultiple::Reconstruct()
{
  string outputPrefixForSlice;
  char c_string[100];
  // Necesito instanciar un Mlem2d 
  Mlem2d* mlem2d;
  // Imagen de un slice:
  SizeImage sizeImage = this->initialEstimate->getSize();
  sizeImage.nPixelsZ = 1;
 
  Image* slice = new Image(sizeImage);
  // Como es una clase derivada mlem2d puedo cargar el sinograma2d correspondiente y llamar a reconstruct.
  for(int i = 0; i < inputProjection->getNumSinograms(); i++)
  {
    // Mensaje:
    cout << "Reconstrucción del slice " << i << endl;
    // Inicializo en uno la imagen inicial:
    slice->fillConstant(1);
    // Instancio mlem2d para el slice i. Le paso los mismos parámetros solo cambio los nombres de salida:
    sprintf(c_string, "%s_slice_%d", this->outputFilenamePrefix.c_str(), i); 
    outputPrefixForSlice.assign(c_string);   
    // Por practicidad no guardo los datos de cada MLEM.
    //mlem2d = new Mlem2d((Sinogram2D*) this->inputProjection->getSinogram2D(i), slice, this->pathSalida, outputPrefixForSlice, this->numIterations, 0, 0,this->sensitivityImageFromFile, this->forwardprojector, this->backprojector);
    mlem2d = new Mlem2d(this->inputProjection->getSinogram2D(i), slice, this->pathSalida, outputPrefixForSlice, this->numIterations, this->saveIterationInterval, this->saveIntermediateProjectionAndBackprojectedImage,this->sensitivityImageFromFile, this->forwardprojector, this->backprojector);
    // Set the multiplcative if its enabled:
    if(enableMultiplicativeTerm)
      mlem2d->setMultiplicativeProjection(this->multiplicativeProjection->getSinogram2D(i));
    // Set the additive if its enabled:
    if(enableAdditiveTerm)
      mlem2d->setAdditiveProjection(this->additiveProjection->getSinogram2D(i));
    mlem2d->Reconstruct();
    // Con el slice ya reconstruido lo debo copiar al volumen. Ya previamente verifiqué que tenía un sinograma por slice, así que simplemente lo copio:
    reconstructionImage->setSlice(i, mlem2d->getReconstructedImage());
  }
  
  // Normalizo el volumen para que todos los slices tengan la misma ganancia:
  // Saco esto porque no es genérico.
  //this->NormalizeVolume();
  // Una vez que recontruí todo, guardo en interfile la imagen con todos los slices:
  sprintf(c_string, "%s_volFinal", this->outputFilenamePrefix.c_str());
  reconstructionImage->writeInterfile(c_string);
  return true;
}

bool Mlem2dMultiple::NormalizeVolume()
{
  float weightSlice, dist1, dist2, x_mm, y_mm, z_mm;
  for(int i = 0; i < reconstructionImage->getSize().nPixelsZ; i++)
  {
    reconstructionImage->getPixelGeomCoord(0,0,i,&x_mm,&y_mm,&z_mm);
    // Si las coordenadas axiales irían entre -reconstructionImage->getFovHeight()/2 y-reconstructionImage->getFovHeight()/2
//     dist1 = abs(z_mm-reconstructionImage->getFovHeight()/2);
//     dist2 = abs(reconstructionImage->getFovHeight()/2-z_mm);
    // Pero por ahora va desde cero a FovHeight.
    dist1 = fabs(z_mm-0);
    dist2 = fabs(reconstructionImage->getFovHeight()-z_mm);
    weightSlice = reconstructionImage->getSize().sizePixelZ_mm/(2*min(dist1,dist2));
    for(int j = 0; j < reconstructionImage->getSize().nPixelsX; j++)
    {
      for(int k = 0; k < reconstructionImage->getSize().nPixelsY; k++)
      {
	reconstructionImage->setPixelValue(j, k, i, reconstructionImage->getPixelValue(j,k,i)*weightSlice);
      }
    }
  }
  return true;
}
