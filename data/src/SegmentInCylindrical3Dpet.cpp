/**
	\file SegmentInCylindrical3Dpet.cpp
	\brief Archivo que contiene la implementación de la clase SegmentInCylindrical3Dpet.

	Este archivo define la clase SegmentInCylindrical3Dpet. 
	\todo 
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2010.10.01
	\version 1.0.0
*/

#include <SegmentInCylindrical3Dpet.h>


using namespace::std;
//using namespace::iostream;

SegmentInCylindrical3Dpet::SegmentInCylindrical3Dpet(int nProj, int nR, int nRings, float rFov_mm, float zFov_mm, int rScanner_mm, 
	  int nSinograms, int nMinRingDiff, int nMaxRingDiff, bool initSinos):Segment(nProj, nR, nRings, rFov_mm, zFov_mm, 
	  nSinograms, nMinRingDiff, nMaxRingDiff)
{
  numSinograms = nSinograms;
  radioScanner = rScanner_mm;
  minRingDiff = nMinRingDiff;
  maxRingDiff = nMaxRingDiff;
  sinograms2D = NULL;
  if(initSinos)
    initSinograms(nProj, nR, rFov_mm, zFov_mm);
}

/// Constructor de copia
SegmentInCylindrical3Dpet::SegmentInCylindrical3Dpet(SegmentInCylindrical3Dpet* srcSegment):Segment((Segment*)srcSegment)
{
  // Las propiedades básicas fueron copiadas en segment.
  sinograms2D = NULL;
  this->radioScanner = srcSegment->radioScanner;
  this->initSinogramsFromSegment(srcSegment);
}

/// Destructor de la clase Segmento.
SegmentInCylindrical3Dpet::~SegmentInCylindrical3Dpet()
{
  for(int i = 0; i < numSinograms; i++)
  {
    delete sinograms2D[i];
  }
  free(sinograms2D);
}

void SegmentInCylindrical3Dpet::initSinograms(int nProj, int nR, float rFov_mm, float zFov_mm)
{
  /// Instancio los sinogramas 2D
  sinograms2D = new Sinogram2DinCylindrical3Dpet*[numSinograms];
  for(int i = 0; i < numSinograms; i++)
  {
    sinograms2D[i] = new Sinogram2DinCylindrical3Dpet(nProj, nR, rFov_mm, this->radioScanner);
  }
}

void SegmentInCylindrical3Dpet::initSinogramsFromSegment(Segment* srcSegment)
{
  /// Instancio los sinogramas 2D
  this->sinograms2D = new Sinogram2DinCylindrical3Dpet*[numSinograms];
  for(int i = 0; i < numSinograms; i++)
  {
    sinograms2D[i] = new Sinogram2DinCylindrical3Dpet(srcSegment->getSinogram2D(i));
    sinograms2D[i]->copyMultipleRingConfig(srcSegment->getSinogram2D(i));
  }
}
// Asigna un sinograma dentro de la lista del segmento:
bool SegmentInCylindrical3Dpet::setSinogram2D(Sinogram2DinCylindrical3Dpet* sinogram2D, int indexInSegment)
{
  // Primero libero la memoria del sinograma y luego creo uno nuevo:
  delete sinograms2D[indexInSegment];
  sinograms2D[indexInSegment] = new Sinogram2DinCylindrical3Dpet(sinogram2D);
}