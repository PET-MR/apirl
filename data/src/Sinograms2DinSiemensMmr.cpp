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
const float Sinograms2DinSiemensMmr::axialFov_mm = 258.0f;

/// Size of each pixel element.
const float Sinograms2DinSiemensMmr::crystalElementSize_mm = 4.0891f;

/// Width of each rings.
const float Sinograms2DinSiemensMmr::widthRings_mm = 4.03125f; //axialFov_mm / numRings;

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

