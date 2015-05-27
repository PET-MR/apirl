/**
	\file ConeOfResponseWithPenetrationProjector.cpp
	\brief Archivo que contiene la implementación de la clase ConeOfResponseProjector.

	\todo 
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2010.11.04
	\version 1.1.0
*/

#include <Siddon.h>
#include <ConeOfResponseWithPenetrationProjector.h>

const float ConeOfResponseWithPenetrationProjector::attenuationThreshold = 0.95f;

ConeOfResponseWithPenetrationProjector::ConeOfResponseWithPenetrationProjector(int nSamplesOnDetector, int nSamplesOnCollimatorSurf, float linAttCoef_cm)
{
  this->numSamplesOnDetector = nSamplesOnDetector;
  this->numSamplesOnCollimatorSurf = nSamplesOnCollimatorSurf;
  this->numLinesPerCone = nSamplesOnDetector * nSamplesOnDetector;
  this->linearAttenuationCoeficcient_cm = linAttCoef_cm;
  // La calculo a partir de la ecuación I=I0*exp(-mu*l). Como utilizo la atenuación at= (1-I/I0), y el coef
  // de atenuación lineal está en 1/cm, el máximo largo queda dado por log(1-at)/(-mu/10))
  this->lengthInCollimatorThreshold_mm = log(1-attenuationThreshold) / (-this->linearAttenuationCoeficcient_cm/10); 
  
  // Creo un vector con todos los largos de segmentos para cada lor de entrada a un detector. Es independiente
  // del bin, solo tengo que tener en cuenta la cantidad de lors y los offsets con las que se forma cada una de
  // ellas. A REALIZAR
  
}

float ConeOfResponseWithPenetrationProjector::getAttenuationWeight(float lengthSegment_mm)
{
  // Neceito la relación I/I0, que va a ser el factor de atenuación, y es igual exp(-mu*l)
  float attWeight = exp(-this->linearAttenuationCoeficcient_cm * lengthSegment_mm / 10);
  return attWeight;
}

bool ConeOfResponseWithPenetrationProjector::Backproject (Sinogram2Dtgs* InputSinogram, Image* outputImage)
{
	Point2D P1, P2;
	Line2D LOR;
	float lengthInCollimator;
	float attenuationWeight;
	SiddonSegment** MyWeightsList = (SiddonSegment**)malloc(sizeof(SiddonSegment*));
	// Tamaño de la imagen:
	SizeImage sizeImage = outputImage->getSize();
	// Puntero a los píxeles:
	float* ptrPixels = outputImage->getPixelsPtr();
	/// Para el cono de respuesta necesito obtener los distintos puntos sobre el dectector,
	/// y sobre la cara del colimador que me generan todas las LORs que generan el cono de respuesta.
	/// Para esto a partir del diámetro del colimador obtengo el paso entre cada punto, y el punto 
	/// inicial para luego ser usado en el for. Lo mismo hago para la cara exterior del colimador,
	/// que lo tengo que tener en cuenta en toda su dimensión.
	/// A los puntos del detector, les doy un margen de 7 mm porque la detección es menor en esa zona
	/// después se podría incluso medir eso, no es complicado, con tener el largo del segmento que cruza
	/// el detector ya se podría.A los puntos del colimador también le doy un margen basado en obtener
	/// el largo de plomo a cada costado, y restarle la fistancia de umbral. Porque sabemos que sin importar
	/// el largo del colimador, a partir de ese margen ya va a estar muy atenuado el gamma.
	float margenDetector_mm = 7;
	float margenCollimador_mm = (InputSinogram->getWidthCollimator_mm()-InputSinogram->getWidthHoleCollimator_mm())/2-lengthInCollimatorThreshold_mm;
	float stepOnDetector_mm = (InputSinogram->getWidthDetector_mm()-2*margenDetector_mm) /  numSamplesOnDetector;
	float stepOnCollimator_mm = (InputSinogram->getWidthCollimator_mm()-2*margenCollimador_mm)  /  numSamplesOnCollimatorSurf;
	float firstPointOnDetector_mm = -(InputSinogram->getWidthDetector_mm()/2-margenDetector_mm) + stepOnDetector_mm /2;
	float lastPointOnDetector_mm = (InputSinogram->getWidthDetector_mm()/2-margenDetector_mm) - stepOnDetector_mm /2;
	float firstPointOnCollimatorSurf_mm = -(InputSinogram->getWidthCollimator_mm()/2-margenCollimador_mm) + stepOnDetector_mm /2;
	float lastPointOnCollimatorSurf_mm = (InputSinogram->getWidthCollimator_mm()/2-margenCollimador_mm) - stepOnDetector_mm /2;
	
	for(int i = 0; i < InputSinogram->getNumProj(); i++)
	{
		for(int j = 0; j < InputSinogram->getNumR(); j++)
		{
		  for(float offsetDetector = firstPointOnDetector_mm; offsetDetector <= lastPointOnDetector_mm; offsetDetector+=stepOnDetector_mm)
		  {
			for(float offsetCaraColimador = firstPointOnCollimatorSurf_mm; offsetCaraColimador <= lastPointOnCollimatorSurf_mm; offsetCaraColimador+=stepOnCollimator_mm)
			{
			  InputSinogram->getPointsFromTgsLor(i, j, offsetDetector, offsetCaraColimador, &P1, &P2);
			  LOR.P0 = P1;
			  LOR.Vx = P2.X - P1.X;
			  LOR.Vy = P2.Y - P1.Y;
			  // Then I look for the intersection between the 3D LOR and the lines that
			  // delimits the voxels
			  // Siddon
			  InputSinogram->getFovLimits(LOR, &P1, &P2);
			  
			  // Obtengo el largo del segmento a través del colimador:
			  lengthInCollimator = InputSinogram->getSegmentLengthInCollimator(offsetDetector, offsetCaraColimador);
			  // Si el segmento es más largo que el de umbral, no hago el procesamiento. De lo contrario, lo realizo
			  // y luego aplico el facto de atenuación.
			  if(lengthInCollimator < lengthInCollimatorThreshold_mm)
			  {
				int LengthList;
				attenuationWeight = getAttenuationWeight(lengthInCollimator);
				Siddon(LOR, outputImage, MyWeightsList, &LengthList,1);
				for(int l = 0; l < LengthList; l++)
				{
					// for every element of the systema matrix different from zero,we do
					// the sum(Aij*bi/Projected) for every i
					if((MyWeightsList[0][l].IndexY>=0) && (MyWeightsList[0][l].IndexX>=0)&&(MyWeightsList[0][l].IndexY<sizeImage.nPixelsY) && (MyWeightsList[0][l].IndexX<sizeImage.nPixelsX))
					{
					  ptrPixels[MyWeightsList[0][l].IndexY * sizeImage.nPixelsX + MyWeightsList[0][l].IndexX] +=
						attenuationWeight * MyWeightsList[0][l].Segment * InputSinogram->getSinogramBin(i,j);
					  if( ptrPixels[MyWeightsList[0][l].IndexY * sizeImage.nPixelsX + MyWeightsList[0][l].IndexX] == numeric_limits<float>::infinity())
					  {
						break;
					  }
					}
				  
				}
				// Now I have my estimated projection for LOR i
				if(LengthList!=0)	free(MyWeightsList[0]);
			  }
			}
		  }
		}
	}
	return true;
}

/// Sobrecarga que realiza la Backprojection de InputSinogram/EstimatedSinogram
bool ConeOfResponseWithPenetrationProjector::DivideAndBackproject (Sinogram2Dtgs* InputSinogram, Sinogram2Dtgs* EstimatedSinogram, Image* outputImage)
{
	Point2D P1, P2;
	Line2D LOR;
	float lengthInCollimator;
	float attenuationWeight;
	SiddonSegment** MyWeightsList = (SiddonSegment**)malloc(sizeof(SiddonSegment*));
	// Tamaño de la imagen:
	SizeImage sizeImage = outputImage->getSize();
	// Puntero a los píxeles:
	float* ptrPixels = outputImage->getPixelsPtr();
	/// Para el cono de respuesta necesito obtener los distintos puntos sobre el dectector,
	/// y sobre la cara del colimador que me generan todas las LORs que generan el cono de respuesta.
	/// Para esto a partir del diámetro del colimador obtengo el paso entre cada punto, y el punto 
	/// inicial para luego ser usado en el for. Lo mismo hago para la cara exterior del colimador,
	/// que lo tengo que tener en cuenta en toda su dimensión.
	/// A los puntos del detector, les doy un margen de 7 mm porque la detección es menor en esa zona
	/// después se podría incluso medir eso, no es complicado, con tener el largo del segmento que cruza
	/// el detector ya se podría.A los puntos del colimador también le doy un margen basado en obtener
	/// el largo de plomo a cada costado, y restarle la fistancia de umbral. Porque sabemos que sin importar
	/// el largo del colimador, a partir de ese margen ya va a estar muy atenuado el gamma.
	float margenDetector_mm = 7;
	float margenCollimador_mm = (InputSinogram->getWidthCollimator_mm()-InputSinogram->getWidthHoleCollimator_mm())/2-lengthInCollimatorThreshold_mm;
	float stepOnDetector_mm = (InputSinogram->getWidthDetector_mm()-2*margenDetector_mm) /  numSamplesOnDetector;
	float stepOnCollimator_mm = (InputSinogram->getWidthCollimator_mm()-2*margenCollimador_mm)  /  numSamplesOnCollimatorSurf;
	float firstPointOnDetector_mm = -(InputSinogram->getWidthDetector_mm()/2-margenDetector_mm) + stepOnDetector_mm /2;
	float lastPointOnDetector_mm = (InputSinogram->getWidthDetector_mm()/2-margenDetector_mm) - stepOnDetector_mm /2;
	float firstPointOnCollimatorSurf_mm = -(InputSinogram->getWidthCollimator_mm()/2-margenCollimador_mm) + stepOnDetector_mm /2;
	float lastPointOnCollimatorSurf_mm = (InputSinogram->getWidthCollimator_mm()/2-margenCollimador_mm) - stepOnDetector_mm /2;
	
	for(int i = 0; i < InputSinogram->getNumProj(); i++)
	{
	  for(int j = 0; j < InputSinogram->getNumR(); j++)
	  {
		for(float offsetDetector = firstPointOnDetector_mm; offsetDetector <= lastPointOnDetector_mm; offsetDetector+=stepOnDetector_mm)
		{
		  for(float offsetCaraColimador = firstPointOnCollimatorSurf_mm; offsetCaraColimador <= lastPointOnCollimatorSurf_mm; offsetCaraColimador+=stepOnCollimator_mm)
		  {
			InputSinogram->getPointsFromTgsLor(i, j, offsetDetector, offsetCaraColimador, &P1, &P2);
			LOR.P0 = P1;
			LOR.Vx = P2.X - P1.X;
			LOR.Vy = P2.Y - P1.Y;
			// Then I look for the intersection between the 3D LOR and the lines that
			// delimits the voxels
			// Siddon
			// Obtengo el largo del segmento a través del colimador:
			lengthInCollimator = InputSinogram->getSegmentLengthInCollimator(offsetDetector, offsetCaraColimador);
			// Si el segmento es más largo que el de umbral, no hago el procesamiento. De lo contrario, lo realizo
			// y luego aplico el facto de atenuación.
			if(lengthInCollimator < lengthInCollimatorThreshold_mm)
			{
			  int LengthList;
			  attenuationWeight = getAttenuationWeight(lengthInCollimator);
			  Siddon(LOR, outputImage, MyWeightsList, &LengthList,1);
			  for(int l = 0; l < LengthList; l++)
			  {
				  // for every element of the systema matrix different from zero,we do
				  // the sum(Aij*bi/Projected) for every i
				  if((MyWeightsList[0][l].IndexY>=0) && (MyWeightsList[0][l].IndexX>=0)&&(MyWeightsList[0][l].IndexY<sizeImage.nPixelsY) && (MyWeightsList[0][l].IndexX<sizeImage.nPixelsX))
					ptrPixels[MyWeightsList[0][l].IndexY * sizeImage.nPixelsX + MyWeightsList[0][l].IndexX] +=
					  attenuationWeight * MyWeightsList[0][l].Segment * InputSinogram->getSinogramBin(i,j) / EstimatedSinogram->getSinogramBin(i,j);	
			  }
			  // Now I have my estimated projection for LOR i
			  if(LengthList!=0) free(MyWeightsList[0]);
			}
		  }
		}
	  }
	}
	return true;
}

bool ConeOfResponseWithPenetrationProjector::Project (Image* inputImage, Sinogram2Dtgs* outputProjection)
{
	Point2D P1, P2;
	Line2D LOR;
	float lengthInCollimator;
	float attenuationWeight;
	SiddonSegment** MyWeightsList = (SiddonSegment**)malloc(sizeof(SiddonSegment*));
	// Tamaño de la imagen:
	SizeImage sizeImage = inputImage->getSize();
	// Puntero a los píxeles:
	float* ptrPixels = inputImage->getPixelsPtr();
	/// Para el cono de respuesta necesito obtener los distintos puntos sobre el dectector,
	/// y sobre la cara del colimador que me generan todas las LORs que generan el cono de respuesta.
	/// Para esto a partir del diámetro del colimador obtengo el paso entre cada punto, y el punto 
	/// inicial para luego ser usado en el for. Lo mismo hago para la cara exterior del colimador,
	/// que lo tengo que tener en cuenta en toda su dimensión.
	/// A los puntos del detector, les doy un margen de 7 mm porque la detección es menor en esa zona
	/// después se podría incluso medir eso, no es complicado, con tener el largo del segmento que cruza
	/// el detector ya se podría.A los puntos del colimador también le doy un margen basado en obtener
	/// el largo de plomo a cada costado, y restarle la fistancia de umbral. Porque sabemos que sin importar
	/// el largo del colimador, a partir de ese margen ya va a estar muy atenuado el gamma.
	float margenDetector_mm = 7;
	float margenCollimador_mm = (outputProjection->getWidthCollimator_mm()-outputProjection->getWidthHoleCollimator_mm())/2-lengthInCollimatorThreshold_mm;
	float stepOnDetector_mm = (outputProjection->getWidthDetector_mm()-2*margenDetector_mm) /  numSamplesOnDetector;
	float stepOnCollimator_mm = (outputProjection->getWidthCollimator_mm()-2*margenCollimador_mm)  /  numSamplesOnCollimatorSurf;
	float firstPointOnDetector_mm = -(outputProjection->getWidthDetector_mm()/2-margenDetector_mm) + stepOnDetector_mm /2;
	float lastPointOnDetector_mm = (outputProjection->getWidthDetector_mm()/2-margenDetector_mm) - stepOnDetector_mm /2;
	float firstPointOnCollimatorSurf_mm = -(outputProjection->getWidthCollimator_mm()/2-margenCollimador_mm) + stepOnDetector_mm /2;
	float lastPointOnCollimatorSurf_mm = (outputProjection->getWidthCollimator_mm()/2-margenCollimador_mm) - stepOnDetector_mm /2;
	
	for(int i = 0; i < outputProjection->getNumProj(); i++)
	{
	  
	  for(int j = 0; j < outputProjection->getNumR(); j++)
	  {
		outputProjection->setSinogramBin(i,j,0);
		for(float offsetDetector = firstPointOnDetector_mm; offsetDetector <= lastPointOnDetector_mm; offsetDetector+=stepOnDetector_mm)
		{
		  for(float offsetCaraColimador = firstPointOnCollimatorSurf_mm; offsetCaraColimador <= lastPointOnCollimatorSurf_mm; offsetCaraColimador+=stepOnCollimator_mm)
		  {
			outputProjection->getPointsFromTgsLor(i, j, offsetDetector, offsetCaraColimador, &P1, &P2);
			LOR.P0 = P1;
			LOR.Vx = P2.X - P1.X;
			LOR.Vy = P2.Y - P1.Y;
			// Then I look for the intersection between the 3D LOR and the lines that
			// delimits the voxels
			// Siddon
			// Obtengo el largo del segmento a través del colimador:
			lengthInCollimator = outputProjection->getSegmentLengthInCollimator(offsetDetector, offsetCaraColimador);
			// Si el segmento es más largo que el de umbral, no hago el procesamiento. De lo contrario, lo realizo
			// y luego aplico el facto de atenuación.
			if(lengthInCollimator < lengthInCollimatorThreshold_mm)
			{
			  int LengthList;
			  attenuationWeight = getAttenuationWeight(lengthInCollimator);
			  Siddon(LOR, inputImage, MyWeightsList, &LengthList,1);
			  for(int l = 0; l < LengthList; l++)
			  {
				  // for every element of the systema matrix different from zero,we do
				  // the sum(Aij*Xj) for every J
				  if((MyWeightsList[0][l].IndexY>=0) && (MyWeightsList[0][l].IndexX>=0)&&(MyWeightsList[0][l].IndexY<sizeImage.nPixelsY) && (MyWeightsList[0][l].IndexX<sizeImage.nPixelsX))
					outputProjection->incrementSinogramBin(i,j, attenuationWeight*MyWeightsList[0][l].Segment * ptrPixels[MyWeightsList[0][l].IndexY * sizeImage.nPixelsX + MyWeightsList[0][l].IndexX]);  
				  //printf("r:%d phi:%d z1:%d z2:%d x:%d y:%d z:%d w:%f", j, i, indexRing1, indexRing2, MyWeightsList[0][l].IndexX, MyWeightsList[0][l].IndexY, MyWeightsList[0][l].IndexZ, MyWeightsList[0][l].Segment);	
			  
			  }
			  if(LengthList!=0) free(MyWeightsList[0]);
			}
		  }
		}
	  }
	}
	return true;
}

/// Para otros tipos de sinogramas por ahora no existe:
bool ConeOfResponseWithPenetrationProjector::Project (Image*,Sinogram3D*)
{
  return false;
}
bool ConeOfResponseWithPenetrationProjector::Backproject(Sinogram3D*, Image*)
{
  return false;
}
bool ConeOfResponseWithPenetrationProjector::DivideAndBackproject (Sinogram3D* InputSinogram, Sinogram3D* EstimatedSinogram, Image* outputImage)
{
  return false;
}
