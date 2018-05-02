

#include <Sinogram3DSiemensMmr.h>


//using namespace::iostream;
//const struct ScannerParameters Sinogram3DSiemensMmr::scannerParameters;
/** Radio del Scanner en mm. */
const float Sinogram3DSiemensMmr::radioScannerMmr_mm = 656.0f/2.0f;
 	
/** Radio del FOV en mm. */
const float Sinogram3DSiemensMmr::radioFovMmr_mm = 594.0f/2.0f;
 
/** Largo axial del FOV en mm. */
const float Sinogram3DSiemensMmr::axialFovMmr_mm = 260.0f;
 	
/// Size of each pixel element.
const float Sinogram3DSiemensMmr::crystalElementSize_mm = 4.0891f;
 	
/** Length of the crystal element. */
const float Sinogram3DSiemensMmr::crystalElementLength_mm = 20;
 	
/// Mean depth of interaction:
const float Sinogram3DSiemensMmr::meanDOI_mm = 9.6f;
 	
/// Width of each rings.
const float Sinogram3DSiemensMmr::widthRingsMmr_mm = 4.0625f; //axialFov_mm / numRings;



Sinogram3DSiemensMmr::Sinogram3DSiemensMmr(int numProj, int numR, int numRings, int numSegments, int* numSinogramsPerSegment, int* minRingDiffPerSegment, 
					   int* maxRingDiffPerSegment):Sinogram3DCylindricalPet(numProj, numR, numRings, Sinogram3DSiemensMmr::radioFovMmr_mm, Sinogram3DSiemensMmr::axialFovMmr_mm, 
					   Sinogram3DSiemensMmr::radioScannerMmr_mm, numSegments, numSinogramsPerSegment, minRingDiffPerSegment, maxRingDiffPerSegment)
{
    // Override the generic parameters of sinogram 3d with the fixed parameters for the mmr:
    this->radioScanner_mm = Sinogram3DSiemensMmr::radioScannerMmr_mm;
    this->widthRings_mm = Sinogram3DSiemensMmr::widthRingsMmr_mm;
    this->axialFov_mm = Sinogram3DSiemensMmr::axialFovMmr_mm;
    this->radioFov_mm = Sinogram3DSiemensMmr::radioFovMmr_mm;
    // Redefine the axial values to the specific crystal size:
    if (ptrAxialvalues_mm == NULL)
	    ptrAxialvalues_mm = (float*) malloc(sizeof(float)*numRings);

    for(int i = 0; i < numRings; i++)
    {
	    ptrAxialvalues_mm[i] = widthRings_mm/2 + widthRings_mm*i;
    } 
    // Initialize ring config:
    if(!this->initRingConfig(numSinogramsPerSegment))
      printf("Sinogram3DCylindricalPet::Sinogram3DCylindricalPet::initRingConfig error while setting the properties of the sinogram.\n");
}


Sinogram3DSiemensMmr::Sinogram3DSiemensMmr(char* fileHeaderPath):Sinogram3DCylindricalPet(this->radioFov_mm, this->axialFov_mm, this->radioScanner_mm)
{
  // Override the generic parameters of sinogram 3d with the fixed parameters for the mmr:
  this->radioScanner_mm = Sinogram3DSiemensMmr::radioScannerMmr_mm;
  this->widthRings_mm = Sinogram3DSiemensMmr::widthRingsMmr_mm;
  this->axialFov_mm = Sinogram3DSiemensMmr::axialFovMmr_mm;
  this->radioFov_mm = Sinogram3DSiemensMmr::radioFovMmr_mm;
  if(!readFromInterfile(fileHeaderPath, this->radioScanner_mm))
  {
	  cout << "Error reading the sinogram in interfile format." << endl;
	  return;
  }
  // Redefine axial values:
  /*for(int i = 0; i < numRings; i++)
  {
    ptrAxialvalues_mm[i] = widthRings_mm/2 + widthRings_mm*i;
  }*/
}

/// Constructor para copia desde otro objeto sinograma3d
Sinogram3DSiemensMmr::Sinogram3DSiemensMmr(Sinogram3DSiemensMmr* srcSinogram3D):Sinogram3DCylindricalPet(srcSinogram3D,0)
{
	// Override the generic parameters of sinogram 3d with the fixed parameters for the mmr:
	this->radioScanner_mm = Sinogram3DSiemensMmr::radioScannerMmr_mm;
	this->widthRings_mm = Sinogram3DSiemensMmr::widthRingsMmr_mm;
	this->axialFov_mm = Sinogram3DSiemensMmr::axialFovMmr_mm;
	this->radioFov_mm = Sinogram3DSiemensMmr::radioFovMmr_mm;
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
  /*for(int i = 0; i < numSegments; i++)
  {
    delete segments[i];
  }
  
  delete segments;*/
}

// Constructor que genera un nuevo sinograma 3D a aprtir de los subsets.
Sinogram3DSiemensMmr::Sinogram3DSiemensMmr(Sinogram3DSiemensMmr* srcSinogram3D, int indexSubset, int numSubsets):Sinogram3DCylindricalPet(srcSinogram3D)
{
	// Override the generic parameters of sinogram 3d with the fixed parameters for the mmr:
	this->radioScanner_mm = Sinogram3DSiemensMmr::radioScannerMmr_mm;
	this->widthRings_mm = Sinogram3DSiemensMmr::widthRingsMmr_mm;
	this->axialFov_mm = Sinogram3DSiemensMmr::axialFovMmr_mm;
	this->radioFov_mm = Sinogram3DSiemensMmr::radioFovMmr_mm;
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
  // El tamaño del nuevo sinograma sería del mismo que el original, pero con menos ángulos de proyección:
  // Calculo cuantas proyecciones va a tener el subset:
  int numProjSubset = (int)floorf((float)numProj / (float)numSubsets);
  // Siempre calculo por defecto, luego si no dio exacta la división, debo agregar un ángulo a la proyección:
  if((numProj%numSubsets)>indexSubset)
    numProjSubset++;
  Sinogram3DSiemensMmr* sino3dSubset = new Sinogram3DSiemensMmr(numProjSubset, numR, numRings, 
	 numSegments, numSinogramsPerSegment, minRingDiffPerSegment, maxRingDiffPerSegment);
  // Copy the ring config of sinos2d:
  sino3dSubset->CopyRingConfigForEachSinogram(this);
  // Con este constructor ya tengo la memoria inicializada, copio los bins:
  float* fullProjAngles = this->getSegment(0)->getSinogram2D(0)->getAngPtr();
  float* subsetProjAngles = (float*)malloc(sizeof(float)*numProjSubset);
  for(int i = 0; i < numSegments; i++)
  {
    for(int j = 0; j < this->segments[i]->getNumSinograms(); j++)
    {
      for(int k = 0; k < numProjSubset; k ++)
      {
	// indice angulo del sino completo:
	int kAngCompleto = indexSubset + numSubsets*k;
	
	// Initialization of Phi Values
	subsetProjAngles[k] = fullProjAngles[kAngCompleto];
	for(int l = 0; l < numR; l++)
	{
	  sino3dSubset->getSegment(i)->getSinogram2D(j)->setSinogramBin(k,l, this->getSegment(i)->getSinogram2D(j)->getSinogramBin(kAngCompleto,l));
	}
      }
      memcpy(sino3dSubset->getSegment(i)->getSinogram2D(j)->getAngPtr(), subsetProjAngles, sizeof(float)*numProjSubset);
    }
  }
  memcpy(sino3dSubset->getAngPtr(), subsetProjAngles, sizeof(float)*numProjSubset);
  // Copy the R values: 
  for(int i = 0; i < this->getNumSegments(); i++)
  {
    for(int j = 0; j < this->getSegment(i)->getNumSinograms(); j++)
    {
      // Copy the R values, because in the sino3d are arc corrected but in the subset not yet:
      memcpy(sino3dSubset->getSegment(i)->getSinogram2D(j)->getRPtr(), this->getSegment(i)->getSinogram2D(j)->getRPtr(), sizeof(float)*this->getSegment(i)->getSinogram2D(j)->getNumR());
    } 
  }
  // Segemntation fault next line: i didnt  malloc memory in sinogram3d?
  //memcpy(sino3dSubset->getRPtr(), this->getRPtr(), sizeof(float)*this->getNumR());
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
