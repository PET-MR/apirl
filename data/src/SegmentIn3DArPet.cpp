/**
	\file SegmentIn3DArPet.cpp
	\brief Archivo que contiene la implementación de la clase SegmentIn3DArPet.

	Este archivo define la clase SegmentIn3DArPet. 
	\todo 
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2010.10.01
	\version 1.0.0
*/

#include <SegmentIn3DArPet.h>


using namespace::std;
//using namespace::iostream;

SegmentIn3DArPet::SegmentIn3DArPet(int nProj, int nR, int nRings, float rFov_mm, float zFov_mm, 
	  int nSinograms, int nMinRingDiff, int nMaxRingDiff):Segment(nProj, nR, nRings, rFov_mm, zFov_mm, 
	  nSinograms, nMinRingDiff, nMaxRingDiff)
{
  numSinograms = nSinograms;
  minRingDiff = nMinRingDiff;
  maxRingDiff = nMaxRingDiff;
  sinograms2D = NULL;
  initSinograms(nProj, nR, rFov_mm, zFov_mm);
}

/// Constructor de copia
SegmentIn3DArPet::SegmentIn3DArPet(SegmentIn3DArPet* srcSegment):Segment((Segment*) srcSegment)
{
  // Propiedades básicas copiadas en el constructode segment.
  /// Instancio los sinogramas 2D
  this->sinograms2D = new Sinogram2Din3DArPet*[numSinograms];
  this->initSinogramsFromSegment(srcSegment);
}

/// Destructor de la clase Segmento.
SegmentIn3DArPet::~SegmentIn3DArPet()
{
	for(int i = 0; i < numSinograms; i++)
	{
		delete sinograms2D[i];
	}
	free(sinograms2D);
}

void SegmentIn3DArPet::initSinograms(int nProj, int nR, float rFov_mm, float zFov_mm)
{
  /// Instancio los sinogramas 2D
  sinograms2D = new Sinogram2Din3DArPet*[numSinograms];
  for(int i = 0; i < numSinograms; i++)
  {
    sinograms2D[i] = new Sinogram2Din3DArPet(nProj, nR, rFov_mm);
  }
}

void SegmentIn3DArPet::initSinogramsFromSegment(Segment* srcSegment)
{
  for(int i = 0; i < numSinograms; i++)
  {
    sinograms2D[i] = new Sinogram2Din3DArPet(srcSegment->getSinogram2D(i));
  }
}

// Asigna un sinograma dentro de la lista del segmento:
bool SegmentIn3DArPet::setSinogram2D(Sinogram2Din3DArPet* sinogram2D, int indexInSegment)
{
  // Primero libero la memoria del sinograma y luego creo uno nuevo:
  delete sinograms2D[indexInSegment];
  sinograms2D[indexInSegment] = new Sinogram2Din3DArPet(sinogram2D);
}


 bool SegmentIn3DArPet::setSinogram2D(Sinogram2DinCylindrical3Dpet* sinogram2D, int indexInSegment)
 {
   // Primero libero la memoria del sinograma y luego creo uno nuevo en blanco, finalmente copio los datos crudos.
   // Hago esto porque el sinograma 2d se vuelve a asignar sin impotar su tamaño.
  delete sinograms2D[indexInSegment];
  sinograms2D[indexInSegment] = new Sinogram2Din3DArPet(sinogram2D->getNumProj(), sinogram2D->getNumR(), sinogram2D->getRadioFov_mm());
  // Copio la configuración de anillos:
  int numZ = sinogram2D->getNumZ();
  int *ptrListRing1 = (int*)malloc(sizeof(int)*numZ);
  int *ptrListRing2 = (int*)malloc(sizeof(int)*numZ);
  float *ptrListZ1_mm = (float*)malloc(sizeof(float)*numZ);
  float *ptrListZ2_mm = (float*)malloc(sizeof(float)*numZ);
  for(int i = 0; i < numZ; i++)
  {
    ptrListRing1[i] = sinogram2D->getRing1FromList(i);
    ptrListRing2[i] = sinogram2D->getRing2FromList(i);
    ptrListZ1_mm[i] = sinogram2D->getAxialValue1FromList(i);
    ptrListZ2_mm[i] = sinogram2D->getAxialValue2FromList(i);
  }
  sinograms2D[indexInSegment]->setMultipleRingConfig(numZ, ptrListRing1, ptrListRing2, ptrListZ1_mm, ptrListZ2_mm);
  // Finalmente copio los datos crudos:
  float* ptr1 = sinograms2D[indexInSegment]->getSinogramPtr();
  float* ptr2 = sinogram2D->getSinogramPtr();
  memcpy(ptr1, ptr2, sinogram2D->getNumProj()*sinogram2D->getNumR()*sizeof(float));
 }