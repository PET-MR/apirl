

#include <Sinogram3DArPet.h>


//using namespace::iostream;

Sinogram3DArPet::Sinogram3DArPet(char* fileHeaderPath, float rFov_mm, float zFov_mm):Sinogram3D(rFov_mm, zFov_mm)
{
  lengthFromBorderBlindArea_mm = 0;
  minDiffDetectors = 1;
  numSegments = 0;
  // Reemplazo la función que lee el interfile del discovery STE por la genérica:
  if(readFromInterfile(fileHeaderPath))
  {
    return;
  }
}

/// Constructor para copia desde otro objeto sinograma3d
Sinogram3DArPet::Sinogram3DArPet(Sinogram3DArPet* srcSinogram3D): Sinogram3D(srcSinogram3D)
{
  /// Las propiedaddes del objeto fuente al objeto siendo instanciado son copiadas en el constructor de la clase sinogram3d.

  /// Ahora hago la copia de los objetos Segmento
  /// Instancio los sinogramas 2D
  segments = new SegmentIn3DArPet*[numSegments];
  for(int i = 0; i < numSegments; i++)
  {
	  segments[i] = new SegmentIn3DArPet(srcSinogram3D->getSegment(i));
  }
  
  // Propiedades de este sinograma en particular:
  this->setMinDiffDetectors(srcSinogram3D->getMinDiffDetectors());
  this->setLengthOfBlindArea(srcSinogram3D->getLengthOfBlindArea());
}

/// Desctructor
Sinogram3DArPet::~Sinogram3DArPet()
{
	/// Tengo que eliminar toda la memoria pedida!
	/// Libero los segmentos
	for(int i = 0; i < numSegments; i++)
	{
		delete segments[i];
	}
	free(segments);
	free(ptrAxialvalues_mm);
}

// Constructor que genera un nuevo sinograma 3D a aprtir de los subsets. Llama al cosntructor copia de sinogram3D
// y luego asigno los sinogramas de los segmentos.
Sinogram3DArPet::Sinogram3DArPet(Sinogram3DArPet* srcSinogram3D, int indexSubset, int numSubsets):Sinogram3D(srcSinogram3D)
{
  // Genero un nuevo sinograma 3d que sea un subset del sinograma principal. La cantidad de segmentos, y sinogramas
  // por segmento la mantengo. Lo único que cambia es que cada sinograma 2D, lo reduzco en ángulo numSubsets veces.
  // Para esto me quedo solo con los ángulos equiespaciados en numSubsets partiendo desde indexSubsets.
  /// Ahora hago la copia de los objetos Segmento
  /// Instancio los sinogramas 2D
  segments = new SegmentIn3DArPet*[numSegments];
  for(int i = 0; i < numSegments; i++)
  {
    segments[i] = new SegmentIn3DArPet(srcSinogram3D->getSegment(i));
  }
  
  // Hasta ahora tengo una copia del sinograma 3d anterior, cambio los sinogramas 2d de cada segmento con los del subset.
  for(int i = 0; i < numSegments; i++)
  {
    for(int j = 0; j < this->segments[i]->getNumSinograms(); j++)
    {
      Sinogram2Din3DArPet* sino2dCompleto = srcSinogram3D->getSegment(i)->getSinogram2D(j);
      segments[i]->setSinogram2D(new Sinogram2Din3DArPet(sino2dCompleto, indexSubset, numSubsets), j);
    }
  }
  this->setLengthOfBlindArea(srcSinogram3D->getLengthOfBlindArea());
}

Sinogram3D* Sinogram3DArPet::Copy()
{
  Sinogram3DArPet* sino3dcopy = new Sinogram3DArPet(this);
  return (Sinogram3D*)sino3dcopy;
}
	
Sinogram3D* Sinogram3DArPet::getSubset(int indexSubset, int numSubsets)
{
  Sinogram3DArPet* sino3dSubset = new Sinogram3DArPet(this);
  Sinogram2Din3DArPet* auxSino2d;
  // Tengo una copia del sinograma, debo cambiar los sinogramas
  for(int i = 0; i < numSegments; i++)
  {
    for(int j = 0; j < this->segments[i]->getNumSinograms(); j++)
    {
      auxSino2d = new Sinogram2Din3DArPet(this->getSegment(i)->getSinogram2D(j));
      sino3dSubset->getSegment(i)->setSinogram2D(auxSino2d, j);
      sino3dSubset->getSegment(i)->getSinogram2D(j)->setMinDiffDetectors(this->getMinDiffDetectors());
      sino3dSubset->getSegment(i)->getSinogram2D(j)->setBlindLength(this->getLengthOfBlindArea());
      delete auxSino2d;
    }
  }
  return (Sinogram3D*)sino3dSubset;
}

/// Función que inicializa los segmentos.
void Sinogram3DArPet::inicializarSegmentos()
{
  /// Instancio los sinogramas 2D
  /// Instancio los sinogramas 2D
  segments = new SegmentIn3DArPet*[numSegments];
  for(int i = 0; i < numSegments; i++)
  {
    segments[i] = new SegmentIn3DArPet(numProj, numR, numRings, radioFov_mm, axialFov_mm, 
	  numSinogramsPerSegment[i], minRingDiffPerSegment[i], maxRingDiffPerSegment[i]);
  }
}
	
/// Setes un sinograma en un segmento.
void Sinogram3DArPet::setSinogramInSegment(int indexSegment, int indexSino, Sinogram2DinCylindrical3Dpet* sino2d)
{
  segments[indexSegment]->setSinogram2D(sino2d, indexSino);
}

void Sinogram3DArPet::setMinDiffDetectors(float minDiff) 
{
  minDiffDetectors = minDiff;
  for(int i = 0; i < this->getNumSegments(); i++)
  {
    for(int j = 0; j < this->getSegment(i)->getNumSinograms(); j++)
    {
      this->getSegment(i)->getSinogram2D(j)->setMinDiffDetectors(minDiff);
    }
  } 
}

void Sinogram3DArPet::setLengthOfBlindArea(float length_m)
{ 
  lengthFromBorderBlindArea_mm = length_m;
  for(int i = 0; i < this->getNumSegments(); i++)
  {
    for(int j = 0; j < this->getSegment(i)->getNumSinograms(); j++)
    {
      this->getSegment(i)->getSinogram2D(j)->setBlindLength(length_m);
    }
  }
}