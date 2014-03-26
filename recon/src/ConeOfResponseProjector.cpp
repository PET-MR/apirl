/**
	\file ConeOfResponseProjector.cpp
	\brief Archivo que contiene la implementación de la clase ConeOfResponseProjector.

	\todo 
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2010.11.04
	\version 1.1.0
*/

#include <Siddon.h>
#include <ConeOfResponseProjector.h>

ConeOfResponseProjector::ConeOfResponseProjector(int nSamplesOnDetector)
{
  this->numSamplesOnDetector = nSamplesOnDetector;
  this->numLinesPerCone = nSamplesOnDetector * nSamplesOnDetector;
  
}

bool ConeOfResponseProjector::Backproject (Sinogram2Dtgs* InputSinogram, Image* outputImage)
{
	Point2D P1, P2;
	Line2D LOR;
	SiddonSegment** MyWeightsList = (SiddonSegment**)malloc(sizeof(SiddonSegment*));
	// Tamaño de la imagen:
	SizeImage sizeImage = outputImage->getSize();
	// Puntero a los píxeles:
	float* ptrPixels = outputImage->getPixelsPtr();
	/// Para el cono de respuesta necesito obtener los distintos puntos sobre el dectector,
	/// y sobre la cara del colimador que me generan todas las LORs que generan el cono de respuesta.
	/// Para esto a partir del diámetro del colimador obtengo el paso entre cada punto, y el punto 
	/// inicial para luego ser usado en el for.
	float stepOnDetector_mm = InputSinogram->getWidthHoleCollimator_mm() /  numSamplesOnDetector;
	float firstPointOnDetector_mm = -(InputSinogram->getWidthHoleCollimator_mm()/2) + stepOnDetector_mm /2;
	float lastPointOnDetector_mm = (InputSinogram->getWidthHoleCollimator_mm()/2) - stepOnDetector_mm /2;

	for(int i = 0; i < InputSinogram->getNumProj(); i++)
	{
		for(int j = 0; j < InputSinogram->getNumR(); j++)
		{
		  for(float offsetDetector = firstPointOnDetector_mm; offsetDetector <= lastPointOnDetector_mm; offsetDetector+=stepOnDetector_mm)
		  {
			for(float offsetCaraColimador = firstPointOnDetector_mm; offsetCaraColimador <= lastPointOnDetector_mm; offsetCaraColimador+=stepOnDetector_mm)
			{
			  GetPointsFromTgsLor (InputSinogram->getAngValue(i), InputSinogram->getRValue(j),  InputSinogram->getDistCrystalToCenterFov(), InputSinogram->getLengthColimator_mm(), 
								   offsetDetector, offsetCaraColimador, &P1, &P2);
			  LOR.P0 = P1;
			  LOR.Vx = P2.X - P1.X;
			  LOR.Vy = P2.Y - P1.Y;
			  // Then I look for the intersection between the 3D LOR and the lines that
			  // delimits the voxels
			  // Siddon
			  InputSinogram->getFovLimits(LOR, &P1, &P2);
			  
			  unsigned int LengthList;
			  Siddon(LOR, outputImage, MyWeightsList, &LengthList,1);
			  for(int l = 0; l < LengthList; l++)
			  {
				  // for every element of the systema matrix different from zero,we do
				  // the sum(Aij*bi/Projected) for every i
				  if((MyWeightsList[0][l].IndexY>=0) && (MyWeightsList[0][l].IndexX>=0)&&(MyWeightsList[0][l].IndexY<sizeImage.nPixelsY) && (MyWeightsList[0][l].IndexX<sizeImage.nPixelsX))
					ptrPixels[MyWeightsList[0][l].IndexY * sizeImage.nPixelsX + MyWeightsList[0][l].IndexX] +=
					  MyWeightsList[0][l].Segment * InputSinogram->getSinogramBin(i,j);				
			  }
			  // Now I have my estimated projection for LOR i
			  if(LengthList!=0)	free(MyWeightsList[0]);
			}
		  }
		}
	}
	return true;
}

/// Sobrecarga que realiza la Backprojection de InputSinogram/EstimatedSinogram
bool ConeOfResponseProjector::DivideAndBackproject (Sinogram2Dtgs* InputSinogram, Sinogram2Dtgs* EstimatedSinogram, Image* outputImage)
{
	Point2D P1, P2;
	Line2D LOR;
	SiddonSegment** MyWeightsList = (SiddonSegment**)malloc(sizeof(SiddonSegment*));
	// Tamaño de la imagen:
	SizeImage sizeImage = outputImage->getSize();
	// Puntero a los píxeles:
	float* ptrPixels = outputImage->getPixelsPtr();
	/// Para el cono de respuesta necesito obtener los distintos puntos sobre el dectector,
	/// y sobre la cara del colimador que me generan todas las LORs que generan el cono de respuesta.
	/// Para esto a partir del diámetro del colimador obtengo el paso entre cada punto, y el punto 
	/// inicial para luego ser usado en el for.
	float stepOnDetector_mm = InputSinogram->getWidthHoleCollimator_mm() /  numSamplesOnDetector;
	float firstPointOnDetector_mm = -(InputSinogram->getWidthHoleCollimator_mm()/2) + stepOnDetector_mm /2;
	float lastPointOnDetector_mm = (InputSinogram->getWidthHoleCollimator_mm()/2) - stepOnDetector_mm /2;
	
	for(int i = 0; i < InputSinogram->getNumProj(); i++)
	{
	  for(int j = 0; j < InputSinogram->getNumR(); j++)
	  {
		for(float offsetDetector = firstPointOnDetector_mm; offsetDetector <= lastPointOnDetector_mm; offsetDetector+=stepOnDetector_mm)
		{
		  for(float offsetCaraColimador = firstPointOnDetector_mm; offsetCaraColimador <= lastPointOnDetector_mm; offsetCaraColimador+=stepOnDetector_mm)
		  {
			GetPointsFromTgsLor (InputSinogram->getAngValue(i), InputSinogram->getRValue(j),  InputSinogram->getDistCrystalToCenterFov(), InputSinogram->getLengthColimator_mm(), 
								  offsetDetector, offsetCaraColimador, &P1, &P2);
			LOR.P0 = P1;
			LOR.Vx = P2.X - P1.X;
			LOR.Vy = P2.Y - P1.Y;
			// Then I look for the intersection between the 3D LOR and the lines that
			// delimits the voxels
			// Siddon
			
			unsigned int LengthList;
			Siddon(LOR, outputImage, MyWeightsList, &LengthList,1);
			for(unsigned int l = 0; l < LengthList; l++)
			{
				// for every element of the systema matrix different from zero,we do
				// the sum(Aij*bi/Projected) for every i
				if((MyWeightsList[0][l].IndexY>=0) && (MyWeightsList[0][l].IndexX>=0)&&(MyWeightsList[0][l].IndexY<sizeImage.nPixelsY) && (MyWeightsList[0][l].IndexX<sizeImage.nPixelsX))
				  ptrPixels[MyWeightsList[0][l].IndexY * sizeImage.nPixelsX + MyWeightsList[0][l].IndexX] +=
					MyWeightsList[0][l].Segment * InputSinogram->getSinogramBin(i,j) / EstimatedSinogram->getSinogramBin(i,j);	
			}
			// Now I have my estimated projection for LOR i
			if(LengthList!=0) free(MyWeightsList[0]);
		  }
		}
	  }
	}
	return true;
}

bool ConeOfResponseProjector::Project (Image* inputImage, Sinogram2Dtgs* outputProjection)
{
	Point2D P1, P2;
	Line2D LOR;
	SiddonSegment** MyWeightsList = (SiddonSegment**)malloc(sizeof(SiddonSegment*));
	// Tamaño de la imagen:
	SizeImage sizeImage = inputImage->getSize();
	// Puntero a los píxeles:
	float* ptrPixels = inputImage->getPixelsPtr();
	/// Para el cono de respuesta necesito obtener los distintos puntos sobre el dectector,
	/// y sobre la cara del colimador que me generan todas las LORs que generan el cono de respuesta.
	/// Para esto a partir del diámetro del colimador obtengo el paso entre cada punto, y el punto 
	/// inicial para luego ser usado en el for.
	float stepOnDetector_mm = outputProjection->getWidthHoleCollimator_mm() /  numSamplesOnDetector;
	float firstPointOnDetector_mm = -(outputProjection->getWidthHoleCollimator_mm()/2) + stepOnDetector_mm /2;
	float lastPointOnDetector_mm = (outputProjection->getWidthHoleCollimator_mm()/2) - stepOnDetector_mm /2;
	
	for(unsigned int i = 0; i < outputProjection->getNumProj(); i++)
	{  
	  for(unsigned int j = 0; j < outputProjection->getNumR(); j++)
	  {
		outputProjection->setSinogramBin(i,j,0);
		for(float offsetDetector = firstPointOnDetector_mm; offsetDetector <= lastPointOnDetector_mm; offsetDetector+=stepOnDetector_mm)
		{
		  for(float offsetCaraColimador = firstPointOnDetector_mm; offsetCaraColimador <= lastPointOnDetector_mm; offsetCaraColimador+=stepOnDetector_mm)
		  {
			GetPointsFromTgsLor (outputProjection->getAngValue(i), outputProjection->getRValue(j),  outputProjection->getDistCrystalToCenterFov(), outputProjection->getLengthColimator_mm(), 
								  offsetDetector, offsetCaraColimador, &P1, &P2);
			LOR.P0 = P1;
			LOR.Vx = P2.X - P1.X;
			LOR.Vy = P2.Y - P1.Y;
			// Then I look for the intersection between the 3D LOR and the lines that
			// delimits the voxels
			// Siddon
			
			unsigned int LengthList;
			Siddon(LOR, inputImage, MyWeightsList, &LengthList,1);
			for(unsigned int l = 0; l < LengthList; l++)
			{
				// for every element of the systema matrix different from zero,we do
				// the sum(Aij*Xj) for every J
				if((MyWeightsList[0][l].IndexY>=0) && (MyWeightsList[0][l].IndexX>=0)&&(MyWeightsList[0][l].IndexY<sizeImage.nPixelsY) && (MyWeightsList[0][l].IndexX<sizeImage.nPixelsX))
				  outputProjection->incrementSinogramBin(i,j, MyWeightsList[0][l].Segment * 					ptrPixels[MyWeightsList[0][l].IndexY * sizeImage.nPixelsX + MyWeightsList[0][l].IndexX]);  
				//printf("r:%d phi:%d z1:%d z2:%d x:%d y:%d z:%d w:%f", j, i, indexRing1, indexRing2, MyWeightsList[0][l].IndexX, MyWeightsList[0][l].IndexY, MyWeightsList[0][l].IndexZ, MyWeightsList[0][l].Segment);	
			
			}
			if(LengthList!=0) free(MyWeightsList[0]);
		  }
		}
	  }
	}
	return true;
}

/// Para otros tipos de sinogramas por ahora no existe:
bool ConeOfResponseProjector::Project (Image*,Sinogram3D*)
{
  return false;
}
bool ConeOfResponseProjector::Backproject(Sinogram3D*, Image*)
{
  return false;
}
bool ConeOfResponseProjector::DivideAndBackproject (Sinogram3D* InputSinogram, Sinogram3D* EstimatedSinogram, Image* outputImage)
{
  return false;
}
