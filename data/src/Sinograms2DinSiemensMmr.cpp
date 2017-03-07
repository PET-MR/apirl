/**
	\file Sinograms2DinSiemensMmr.cpp
	\brief Archivo que contiene la implementación de la clase Sinograms2DinSiemensMmr.

	Este archivo define la clase Sinograms2DinSiemensMmr. 
	\todo 
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2015.07.10
	\version 1.0.0
*/

#include <Sinograms2DinSiemensMmr.h>


using namespace::std;
//using namespace::iostream;
/** Radio del Scanner en mm. */
const float Sinograms2DinSiemensMmr::radioScanner_mm = 656.0f/2.0f;

/** Radio del FOV en mm. */
const float Sinograms2DinSiemensMmr::radioFov_mm = 594.0f/2.0f;

/** Largo axial del FOV en mm. */
const float Sinograms2DinSiemensMmr::axialFov_mm = 257.96875f;

/// Size of each pixel element.
const float Sinograms2DinSiemensMmr::crystalElementSize_mm = 4.0891f;

/// Width of each rings.
const float Sinograms2DinSiemensMmr::widthRings_mm = 4.0625f; // 4 mm crystal + a gap between crystal rings of 0.0625 (in the e7 documentation says 0.40625, but that must be wrong)


/// Constructor de copia
Sinograms2DinSiemensMmr::Sinograms2DinSiemensMmr(Sinograms2DinSiemensMmr* srcSino):Sinograms2DmultiSlice((Sinograms2DmultiSlice*) srcSino)
{
  /// Instancio los sinogramas 2D
  initSinograms();
  CopyBins((Sinograms2DmultiSlice*)srcSino);
}

/// Constructor desde un archivo:
Sinograms2DinSiemensMmr::Sinograms2DinSiemensMmr(string fileName):Sinograms2DmultiSlice(fileName, radioFov_mm, axialFov_mm)
{
  this->readFromInterfile(fileName);
  initParameters();
//   float lr;
//   // The r value is non linear in the sinogram, because each bin represent one detector element and
//   // with the curve of the cylindrical scanner the distance r to the center axis increases with the cos of the bin.
//   for(int j = 0; j < numR; j++)
//   {
//     // ptrRvalues initialization is necesary just one time
//     // 1) Get the length on the cylindrical surface for each bin (from x=0 to the center of the crystal element):
//     lr = (Sinogram2DinSiemensMmr::binSize_mm/2 + Sinogram2DinSiemensMmr::binSize_mm*(j-(float)(numR/2)));
//     // 2) Now I get the x coordinate for that r.
//     ptrRvalues_mm[j] = (radioScanner_mm + Sinogram2DinSiemensMmr::meanDOI_mm* cos(lr/radioScanner_mm)) * sin(lr/radioScanner_mm);
//   }
}


/// Destructor de la clase Segmento.
Sinograms2DinSiemensMmr::~Sinograms2DinSiemensMmr()
{
  for(int i = 0; i < numSinograms; i++)
  {
	  delete sinograms2D[i];
  }
  free(sinograms2D);
}

void Sinograms2DinSiemensMmr::initSinograms()
{
  /// Instancio los sinogramas 2D
  sinograms2D = new Sinogram2DinSiemensMmr*[numSinograms];
  for(int i = 0; i < numSinograms; i++)
  {
    sinograms2D[i] = new Sinogram2DinSiemensMmr(numProj, numR);
  }
}

