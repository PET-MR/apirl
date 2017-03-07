/**
	\file Sinograms2DinCylindrical3Dpet.cpp
	\brief Archivo que contiene la implementación de la clase Sinograms2DinCylindrical3Dpet.

	Este archivo define la clase Sinograms2DinCylindrical3Dpet. 
	\todo 
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2012.10.01
	\version 1.0.0
*/

#include <Sinograms2DinCylindrical3Dpet.h>


using namespace::std;
//using namespace::iostream;

Sinograms2DinCylindrical3Dpet::Sinograms2DinCylindrical3Dpet(int nProj, int nR, float rFov_mm, float zFov_mm, float rScanner_mm, 
	  int nSinograms) : Sinograms2DmultiSlice(nProj, nR, rFov_mm, zFov_mm, nSinograms)
{
  this->rScanner_mm = rScanner_mm;
  /// Instancio los sinogramas 2D
  initSinograms();
  initParameters();
}

/// Constructor de copia
Sinograms2DinCylindrical3Dpet::Sinograms2DinCylindrical3Dpet(Sinograms2DinCylindrical3Dpet* srcSino):Sinograms2DmultiSlice((Sinograms2DmultiSlice*) srcSino)
{
  this->rScanner_mm = srcSino->rScanner_mm;
  /// Instancio los sinogramas 2D
  initSinograms();
  CopyBins((Sinograms2DmultiSlice*)srcSino);
}

/// Constructor desde un archivo:
Sinograms2DinCylindrical3Dpet::Sinograms2DinCylindrical3Dpet(string fileName, float rF_mm, float zF_mm, float rSca_mm):Sinograms2DmultiSlice(fileName, rF_mm, zF_mm)
{
  this->rScanner_mm = rSca_mm;
  this->readFromInterfile(fileName);
  initParameters();
}

/// Destructor de la clase Segmento.
Sinograms2DinCylindrical3Dpet::~Sinograms2DinCylindrical3Dpet()
{
  for(int i = 0; i < numSinograms; i++)
  {
	  delete sinograms2D[i];
  }
  free(sinograms2D);
}

void Sinograms2DinCylindrical3Dpet::initSinograms()
{
  /// Instancio los sinogramas 2D
  sinograms2D = new Sinogram2DinCylindrical3Dpet*[numSinograms];
  for(int i = 0; i < numSinograms; i++)
  {
    sinograms2D[i] = new Sinogram2DinCylindrical3Dpet(numProj, numR, radioFov_mm, rScanner_mm);
  }
}

void Sinograms2DinCylindrical3Dpet::setRadioScannerInSinograms()
{
  for(int i = 0; i < numSinograms; i++)
  {
    sinograms2D[i]->setRadioScanner(this->rScanner_mm);
  }
}