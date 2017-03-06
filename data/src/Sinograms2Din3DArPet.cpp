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

#include <Sinograms2Din3DArPet.h>


using namespace::std;
//using namespace::iostream;

Sinograms2Din3DArPet::Sinograms2Din3DArPet(int nProj, int nR, float rFov_mm, float zFov_mm, int nSinograms) : Sinograms2DmultiSlice(nProj, nR, rFov_mm, zFov_mm, nSinograms)
{
  /// Instancio los sinogramas 2D
  lengthFromBorderBlindArea_mm = 0;
  minDiffDetectors = 1;
  initSinograms();
}

/// Constructor de copia
Sinograms2Din3DArPet::Sinograms2Din3DArPet(Sinograms2Din3DArPet* srcSino) : Sinograms2DmultiSlice((Sinograms2DmultiSlice*)srcSino)
{
  /// Instancio los sinogramas 2D
  initSinograms();
  CopyBins((Sinograms2DmultiSlice*)srcSino);
  this->setLengthOfBlindArea(srcSino->getLengthOfBlindArea());
  this->setMinDiffDetectors(srcSino->getMinDiffDetectors());
}

/// Constructor desde un archivo:
Sinograms2Din3DArPet::Sinograms2Din3DArPet(string fileName, float rF_mm, float zF_mm):Sinograms2DmultiSlice(fileName, rF_mm, zF_mm)
{  
  this->readFromInterfile(fileName);
  initParameters();
}

/// Destructor de la clase Sinograms2Din3DArPet.
Sinograms2Din3DArPet::~Sinograms2Din3DArPet()
{
	for(int i = 0; i < numSinograms; i++)
	{
		delete sinograms2D[i];
	}
	free(sinograms2D);
}
void Sinograms2Din3DArPet::initSinograms()
{
  /*if(sinograms2D != NULL)
  {
    for(int i = 0; i < numSinograms; i++)
    {
      if(sinograms2D[i] != NULL)
	free(sinograms2D[i]);
    }
    free(sinograms2D);
  }*/
  /// Instancio los sinogramas 2D
  sinograms2D = new Sinogram2Din3DArPet*[numSinograms];
  for(int i = 0; i < numSinograms; i++)
  {
    sinograms2D[i] = new Sinogram2Din3DArPet(numProj, numR, radioFov_mm);
  }
  
}

Sinogram2D* Sinograms2Din3DArPet::getSinogram2DCopy(int indexSinogram2D)
{
  Sinogram2Din3DArPet* sino2d = new Sinogram2Din3DArPet(sinograms2D[indexSinogram2D]);
  return (Sinogram2D*)sino2d; 
  
};

void Sinograms2Din3DArPet::setMinDiffDetectors(float minDiff) 
{
  minDiffDetectors = minDiff;
  for(int i = 0; i < numSinograms; i++)
  {
    this->sinograms2D[i]->setMinDiffDetectors(minDiff);
  } 
}

void Sinograms2Din3DArPet::setLengthOfBlindArea(float length_m)
{ 
  lengthFromBorderBlindArea_mm = length_m;
  for(int i = 0; i < numSinograms; i++)
  {
    this->sinograms2D[i]->setBlindLength(length_m);
  }
}
  
