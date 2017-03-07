/**
	\file Segment.cpp
	\brief Archivo que contiene la implementación de la clase Segment.

	Este archivo define la clase Segment. 
	\todo 
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2010.10.01
	\version 1.0.0
*/

#include <Segment.h>


using namespace::std;
//using namespace::iostream;

Segment::Segment(int nProj, int nR, int nRings, float rFov_mm, float zFov_mm, 
	  int nSinograms, int nMinRingDiff, int nMaxRingDiff)
{
  numSinograms = nSinograms;
  /// Instancio los sinogramas 2D
  //initSinograms(nProj, nR, rFov_mm, zFov_mm);
  minRingDiff = nMinRingDiff;
  maxRingDiff = nMaxRingDiff;
}

/// Constructor de copia
Segment::Segment(Segment* srcSegment)
{
  this->numSinograms = srcSegment->numSinograms;
  this->minRingDiff = srcSegment->getMinRingDiff();
  this->maxRingDiff = srcSegment->getMaxRingDiff();
  /// Instancio los sinogramas 2D
  //initSinogramsFromSegment(srcSegment);
}
/// Destructor de la clase Segmento.
Segment::~Segment()
{

}
