/**
	\file SegmentInSiemensMmr.cpp
	\brief Archivo que contiene la implementación de la clase SegmentInSiemensMmr.

	Este archivo define la clase SegmentInSiemensMmr, que solo hace un override de algunos métodos
	de SegmentInCylindrical3Dpet, especialmente la inicialización de sinogramas. 
	\todo 
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2010.10.01
	\version 1.0.0
*/

#include <SegmentInSiemensMmr.h>


using namespace::std;
//using namespace::iostream;

SegmentInSiemensMmr::SegmentInSiemensMmr(int nProj, int nR, int nRings, float rFov_mm, float zFov_mm, float rScanner_mm, 
	  int nSinograms, int nMinRingDiff, int nMaxRingDiff):SegmentInCylindrical3Dpet(nProj, nR, nRings, rFov_mm, zFov_mm, rScanner_mm, 
	  nSinograms, nMinRingDiff, nMaxRingDiff, 0)	// 0 to not init memory of sinograms.
{
  numSinograms = nSinograms;
  radioScanner = rScanner_mm;
  minRingDiff = nMinRingDiff;
  maxRingDiff = nMaxRingDiff;
  initSinograms(nProj, nR, rFov_mm, zFov_mm);
}

/// Constructor de copia
SegmentInSiemensMmr::SegmentInSiemensMmr(SegmentInSiemensMmr* srcSegment):SegmentInCylindrical3Dpet((SegmentInCylindrical3Dpet*)srcSegment)
{
  // Idem sinogram 3d cylindrico, debería llamar al inti singoram de acá.
}

/// Destructor de la clase Segmento.
SegmentInSiemensMmr::~SegmentInSiemensMmr()
{
  // It's done by the SegmentInCylindrical3Dpet class.
  /*for(int i = 0; i < numSinograms; i++)
  {
    delete sinograms2D[i];
  }
  delete sinograms2D;*/
}

void SegmentInSiemensMmr::initSinograms(int nProj, int nR, float rFov_mm, float zFov_mm)
{
  /// Instancio los sinogramas 2D
  sinograms2D = (Sinogram2DinCylindrical3Dpet**) new Sinogram2DinSiemensMmr*[this->getNumSinograms()];
  for(int i = 0; i < numSinograms; i++)
  {
    sinograms2D[i] = new Sinogram2DinSiemensMmr(nProj, nR);
  }
}

void SegmentInSiemensMmr::initSinogramsFromSegment(Segment* srcSegment)
{
  /// If the sinogram2d has already allocated memory, we free it and then ask it
  /// again for the same.
  if ((numSinograms != srcSegment->getNumSinograms())||(maxRingDiff != srcSegment->getMaxRingDiff())||(minRingDiff != srcSegment->getMinRingDiff()))
  {
    perror("Sinogram2DinSiemensMmr::initSinogramsFromSegment: trying to init sinograms with a different segment (numSinogras or maxRingDiff or minRingDiff)\n");
    exit(1);
  }
  if(this->sinograms2D == NULL)
    this->sinograms2D = (Sinogram2DinCylindrical3Dpet**) new Sinogram2DinSiemensMmr*[numSinograms];
  for(int i = 0; i < numSinograms; i++)
  {
    // Free actual sinogram:
    if(sinograms2D[i] != NULL)
      delete sinograms2D[i];
    sinograms2D[i] = (Sinogram2DinCylindrical3Dpet*) new Sinogram2DinSiemensMmr((Sinogram2DinSiemensMmr*)srcSegment->getSinogram2D(i));
  }
}
// Asigna un sinograma dentro de la lista del segmento:
bool SegmentInSiemensMmr::setSinogram2D(Sinogram2DinSiemensMmr* sinogram2D, int indexInSegment)
{
  // Primero libero la memoria del sinograma y luego creo uno nuevo:
  delete sinograms2D[indexInSegment];
  sinograms2D[indexInSegment] = new Sinogram2DinSiemensMmr(sinogram2D);
  return true;
}