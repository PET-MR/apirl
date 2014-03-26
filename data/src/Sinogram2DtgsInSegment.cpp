/**
	\file Sinogram2DtgsInSegment.cpp
	\brief Archivo que contiene la implementación de la clase Sinogram2DtgsInSegment.

	Este archivo define la clase Sinogram2tgsInSegment. 
	Al ser un sinograma de SPECT tiene proyecciones de 0 a 360º. Además le agrego propiedades
	que hacen a este tipo de sinograma, por ejemplo el largo y ancho del colimador, para poder
	obtener el Cone of Response para cada lor. Ahora también se incluye el ancho total del colimador
	para poder modelar la penetración a través del plomo.
	\todo 
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2012.01.09
	\version 1.1.0
*/

#include <Sinogram2DtgsInSegment.h>


Sinogram2DtgsInSegment::Sinogram2DtgsInSegment()
{
  /* Para un sinograma genérico los ángulos de proyecciones van de 0 a 180º. */
  minAng_deg = 0;
  maxAng_deg = 360;
}

Sinogram2DtgsInSegment::Sinogram2DtgsInSegment(unsigned int nProj, unsigned int nR, float rFov_mm, float dCrystalToCenterFov, float lColimator_mm, float wCollimator_mm, float wHoleCollimator_mm, float wSegment_mm) :
Sinogram2Dtgs(nProj, nR, rFov_mm, dCrystalToCenterFov, lColimator_mm, wCollimator_mm, wHoleCollimator_mm)
{
  widthSegment_mm = wSegment_mm;
}

/// Constructor de Copia
Sinogram2DtgsInSegment::Sinogram2DtgsInSegment(const Sinogram2DtgsInSegment* srcSinogram2D) : Sinogram2Dtgs((Sinogram2Dtgs*) srcSinogram2D)
{
  // Sigo con la copia de los parámetros propios de Sinogram2DtgsInSegment:
  this->widthSegment_mm = srcSinogram2D->widthSegment_mm;
}

void Sinogram2DtgsInSegment::setGeometricParameters(float rFov_mm, float dCrystalToCenterFov, float lColimator_mm, float wCollimator_mm, float wHoleCollimator_mm, float wSegment_mm)
{
  this->radioFov_mm = rFov_mm;
  this->distCrystalToCenterFov = dCrystalToCenterFov;
  this->lengthColimator_mm = lColimator_mm;
  this->widthCollimator_mm = wCollimator_mm;
  this->widthHoleCollimator_mm = wHoleCollimator_mm;
  float rStep = (2 * radioFov_mm) / numR;
  for(unsigned int j = 0; j < numR; j++)
  {	  
	ptrRvalues_mm[j] = rStep/2 + j * rStep - radioFov_mm;
  }
  this->widthSegment_mm = wSegment_mm;
}
