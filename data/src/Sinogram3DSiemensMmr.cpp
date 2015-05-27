

#include <Sinogram3DSiemensMmr.h>


//using namespace::iostream;

/** Radio del Scanner en mm. */
const float Sinogram3DSiemensMmr::radioScanner_mm = 656.0f/2.0f;

/** Radio del FOV en mm. */
const float Sinogram3DSiemensMmr::radioFov_mm = 594.0f/2.0f;

/** Largo axial del FOV en mm. */
const float Sinogram3DSiemensMmr::axialFov_mm = 258.0f;

/// Size of each pixel element.
const float Sinogram3DSiemensMmr::crystalElementSize_mm = 4.0891f;

/// Width of each rings.
const float Sinogram3DSiemensMmr::widthRings_mm = 4.0312f; //axialFov_mm / numRings;

Sinogram3DSiemensMmr::Sinogram3DSiemensMmr(char* fileHeaderPath):Sinogram3DCylindricalPet(this->radioFov_mm, this->axialFov_mm, this->radioScanner_mm)
{
  if(readFromInterfile(fileHeaderPath, this->radioScanner_mm))
  {
    return;
  }
}

/// Constructor para copia desde otro objeto sinograma3d
Sinogram3DSiemensMmr::Sinogram3DSiemensMmr(Sinogram3DSiemensMmr* srcSinogram3D):Sinogram3DCylindricalPet(srcSinogram3D,0)
{
  // Como uso el constructor que inicializa los segemntos, lo hago acá.
  inicializarSegmentos();
  // Copy the content of all the sinograms.
  CopyAllBinsFrom(srcSinogram3D);
  // Copy ring configurations:
  CopyRingConfigForEachSinogram(srcSinogram3D);
}

/// Desctructor
Sinogram3DSiemensMmr::~Sinogram3DSiemensMmr()
{
  // Destructor fo Sinogram3DCylindricalPet frees all the segments memory.
}

// Constructor que genera un nuevo sinograma 3D a aprtir de los subsets.
Sinogram3DSiemensMmr::Sinogram3DSiemensMmr(Sinogram3DSiemensMmr* srcSinogram3D, int indexSubset, int numSubsets):Sinogram3DCylindricalPet(srcSinogram3D)
{
  // Genero un nuevo sinograma 3d que sea un subset del sinograma principal. La cantidad de segmentos, y sinogramas
  // por segmento la mantengo. Lo único que cambia es que cada sinograma 2D, lo reduzco en ángulo numSubsets veces.
  // Para esto me quedo solo con los ángulos equiespaciados en numSubsets partiendo desde indexSubsets.
  /// Copio todas las propiedaddes del objeto fuente al objeto siendo instanciado.
  /// Ahora hago la copia de los objetos Segmento
  inicializarSegmentos();
  
  // Hasta ahora tengo una copia del sinograma 3d anterior, cambio los sinogramas 2d de cada segmento con los del subset.
  for(int i = 0; i < numSegments; i++)
  {
    for(int j = 0; j < srcSinogram3D->getSegment(i)->getNumSinograms(); j++)
    {
      Sinogram2DinSiemensMmr* sino2dCompleto = (Sinogram2DinSiemensMmr*)srcSinogram3D->getSegment(i)->getSinogram2D(j);
      segments[i]->setSinogram2D(new Sinogram2DinSiemensMmr(sino2dCompleto, indexSubset, numSubsets), j);
    }
  }
}

Sinogram3D* Sinogram3DSiemensMmr::Copy()
{
  Sinogram3DSiemensMmr* sino3dcopy = new Sinogram3DSiemensMmr(this);
  return (Sinogram3D*)sino3dcopy;
}

int Sinogram3DSiemensMmr::CopyAllBinsFrom(Sinogram3D* srcSinogram3D)
{
  int numBins = 0;
  for(int i = 0; i < this->getNumSegments(); i++)
  {
    for(int j = 0; j < this->getSegment(i)->getNumSinograms(); j++)
    {
      float *ptrSrc, *ptrDst;
      ptrDst = this->getSegment(i)->getSinogram2D(j)->getSinogramPtr();
      ptrSrc = srcSinogram3D->getSegment(i)->getSinogram2D(j)->getSinogramPtr();
      memcpy(ptrDst, ptrSrc, this->getSegment(i)->getSinogram2D(j)->getNumProj()*this->getSegment(i)->getSinogram2D(j)->getNumR()*sizeof(float));
      numBins += this->getSegment(i)->getSinogram2D(j)->getNumProj()*this->getSegment(i)->getSinogram2D(j)->getNumR();
    } 
  }
  return numBins;
}

Sinogram3D* Sinogram3DSiemensMmr::getSubset(int indexSubset, int numSubsets)
{
  // El tamaño del nuevo sinograma sería del mismo que el original, luego se hace un setSinogram2d para todos los sinos 
  // de los segmentos y debería quedar del tamaño correcto. No se pierde memoria porque setSinogram2D hace un delete primero
  // y luego crea el nuevo.
  Sinogram3DSiemensMmr* sino3dSubset = new Sinogram3DSiemensMmr(this);
  Sinogram2DinSiemensMmr* auxSinos2d;
  // Tengo una copia del sinograma, debo cambiar los sinogramas
  for(int i = 0; i < numSegments; i++)
  {
    for(int j = 0; j < this->segments[i]->getNumSinograms(); j++)
    {
      // Create new sinogram2d with the correct dimensiones:
      auxSinos2d = new Sinogram2DinSiemensMmr((Sinogram2DinSiemensMmr*)this->getSegment(i)->getSinogram2D(j), indexSubset, numSubsets);
      sino3dSubset->getSegment(i)->setSinogram2D(auxSinos2d, j);
      // if debug, printf ang values:
      #ifdef __DEBUG__
	if ((i==0)&&(j==0))
	{
	  printf("Angulos para subset %d de %d: \n", indexSubset, numSubsets);
	  for(int ang = 0; ang < auxSinos2d->getNumProj(); ang++)
	  {
	    printf("%f\t", auxSinos2d->getAngValue(ang));
	  }
	  printf("\n");
	}
      #endif
      // delete the aux sinogram because set sinograms makes a copy:
      delete auxSinos2d;
    }
  }
  return (Sinogram3D*)sino3dSubset;
}

/// Función que inicializa los segmentos.
void Sinogram3DSiemensMmr::inicializarSegmentos()
{
  // First check if there is memory allocated for the segment, and free it in that case:
  if(segments != NULL)
  {
    for(int i = 0; i < numSegments; i++)
    {
      if(segments[i] != NULL)
      {
	delete segments[i];
      }
    }
    delete segments;
  }
  /// Instancio los sinogramas 2D
  segments = (SegmentInCylindrical3Dpet**) new SegmentInSiemensMmr*[numSegments];
  for(int i = 0; i < numSegments; i++)
  {
    segments[i] = new SegmentInSiemensMmr(numProj, numR, numRings, radioFov_mm, axialFov_mm, radioScanner_mm, 
	  numSinogramsPerSegment[i], minRingDiffPerSegment[i], maxRingDiffPerSegment[i]);
  }
}
	
/// Setes un sinograma en un segmento.
void Sinogram3DSiemensMmr::setSinogramInSegment(int indexSegment, int indexSino, Sinogram2DinSiemensMmr* sino2d)
{
  segments[indexSegment]->setSinogram2D(sino2d, indexSino);
}
