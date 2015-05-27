/**
	\file ConeOfResponseProjectorWithAttenuation.cpp
	\brief Archivo que contiene la implementación de la clase ConeOfResponseProjector.

	\todo 
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2010.11.04
	\version 1.2.0
*/

#include <Siddon.h>
#include <ConeOfResponseProjectorWithAttenuation.h>

ConeOfResponseProjectorWithAttenuation::ConeOfResponseProjectorWithAttenuation(int nSamplesOnDetector, Image* attImage)
{
  this->numSamplesOnDetector = nSamplesOnDetector;
  this->numLinesPerCone = nSamplesOnDetector * nSamplesOnDetector;
  this->attenuationMap = new Image(attImage);
}



bool ConeOfResponseProjectorWithAttenuation::Project(Image* inputImage, Sinogram2Dtgs* outputProjection)
{
  Point2D P1, P2;
  Point2D pointOnDetector;
  Line2D LOR;
  float rayLengthInFov_mm;
  float geometricValue, distanceValue;
  float attenuationLength, attenuationFactorEntryLimit, attenuationFactorOutLimit, attenuationWeight;
  float xPixel_mm, yPixel_mm;
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
  
  for(int i = 0; i < outputProjection->getNumProj(); i++)
  {  
	for(int j = 0; j < outputProjection->getNumR(); j++)
	{
	  outputProjection->setSinogramBin(i,j,0);
	  for(float offsetDetector = firstPointOnDetector_mm; offsetDetector <= lastPointOnDetector_mm; offsetDetector+=stepOnDetector_mm)
	  {
		// Necesito la distancia del detector al primer píxel intersectado por siddon, para tenerlo en cuenta
		// en el factor del cuadrado de la distancia. Para esto necesito el punto sobre el detector:
		pointOnDetector = outputProjection->getPointOnDetector(i, j, offsetDetector);
		for(float offsetCaraColimador = firstPointOnDetector_mm; offsetCaraColimador <= lastPointOnDetector_mm; offsetCaraColimador+=stepOnDetector_mm)
		{
		  outputProjection->getPointsFromTgsLor(i, j, offsetDetector, offsetCaraColimador, &P1, &P2);
		  LOR.P0 = P1;
		  LOR.Vx = P2.X - P1.X;
		  LOR.Vy = P2.Y - P1.Y;
		  //printf("P1: %fmm %fmm.\tP2: %fmm %fmm.\n", P1.X, P1.Y, P2.X, P2.Y);
		  // Then I look for the intersection between the 3D LOR and the lines that
		  // delimits the voxels
		  // Siddon
		  
		  int LengthList;
		  rayLengthInFov_mm = Siddon(LOR, inputImage, MyWeightsList, &LengthList,1);
		  geometricValue = 0;
		  // Para la implementación de la atenuación utilizo la implementación propuesta en: 
		  // Gullberg et al, An attenuated projector-backprojector for iterative SPECT reconstruction, Phys Med Biol, 1985, Vol 30, No 8
		  // No tiene ningñun truco en especial, es muy similar a lo que hacía yo originalmente, pero considera un promedio del coeficiente
		  // de atenuación en el píxel (a partir den una integral), en cambio yo usaba el del extremo.
		  attenuationLength = 0;	// En el paper variable ATENL.
		  attenuationFactorEntryLimit = exp(-attenuationLength);	// Peso de atenuación total en el punto de entrada al píxel a procesar.
																// en el paper EX1.
		  attenuationFactorOutLimit = exp(-attenuationLength);	// Peso de atenuación total en el punto de salida al píxel a procesar
																// en el paper EX2.
		  attenuationWeight = 0;	// Peso de atenuación para una lor y píxel dado. Es el factor multiplicativo aplicado a la proyección
									// geométrica.
		  if(rayLengthInFov_mm > 0)
		  {
			if(LengthList!=0)
			{
			  // Obtengo las coordenadas del primer píxel de entrada al fov.
			  inputImage->getPixelGeomCoord(MyWeightsList[0][0].IndexX, MyWeightsList[0][0].IndexY, &xPixel_mm, &yPixel_mm);

			  // Con esas coordenadas y el punto del detector puedo obtener la distancia del detector a la entrada
			  // del fov, que va a ser el valor inicial del valor de distancia, que se usará para aplicar el factor
			  // de ivnerso de la distancia:
			  distanceValue = sqrt((pointOnDetector.X-xPixel_mm) * (pointOnDetector.X-xPixel_mm) +
				(pointOnDetector.Y-yPixel_mm) * (pointOnDetector.Y-yPixel_mm));
			  // Donde entro al FoV, supongo que todo lo anterior fue aire, por lo que la atenuación incial, es
			  // la distancia a la entrada, por el coeficiente de atenuación del aire en 1/mm:
			  attenuationLength = distanceValue * 0.0000097063f;
			  attenuationFactorEntryLimit = exp(-attenuationLength);
			
			  for(int l = 0; l < LengthList; l++)
			  {
				// for every element of the systema matrix different from zero,we do
				// the sum(Aij*Xj) for every J
				if((MyWeightsList[0][l].IndexY>=0) && (MyWeightsList[0][l].IndexX>=0)&&(MyWeightsList[0][l].IndexY<sizeImage.nPixelsY) && (MyWeightsList[0][l].IndexX<sizeImage.nPixelsX))
				{
				  // El geomtricValue va incrementando valor de pixel * siddon.
				  geometricValue = (MyWeightsList[0][l].Segment/sizeImage.sizePixelX_mm) * 					 
					ptrPixels[MyWeightsList[0][l].IndexY * sizeImage.nPixelsX + MyWeightsList[0][l].IndexX];

				  // Para el atenuation value tengo que sumar todos los mu_pixel * siddon:
				  attenuationLength += attenuationMap->getPixelValue(MyWeightsList[0][l].IndexX, MyWeightsList[0][l].IndexY, 0) * MyWeightsList[0][l].Segment; 
				  attenuationFactorOutLimit = exp(-attenuationLength);
				  if((attenuationMap->getPixelValue(MyWeightsList[0][l].IndexX, MyWeightsList[0][l].IndexY, 0)>0)&& (MyWeightsList[0][l].Segment!=0)&&(attenuationFactorEntryLimit!=attenuationFactorOutLimit))
				  {
					// Si el mu de atenuación del píxel es mayor que cero, el factor de atenuación es igual a 
					// (attenuationFactorEntryLimit-attenuationFactorOutLimit/mu_pixel. Esto sale de integrar la atenuación
					// a lo largo del segmento que cruza el píxel:
					//attenuationWeight = (attenuationFactorEntryLimit+attenuationFactorOutLimit)/2;
					attenuationWeight = (attenuationFactorEntryLimit-attenuationFactorOutLimit) / (attenuationMap->getPixelValue(MyWeightsList[0][l].IndexX, MyWeightsList[0][l].IndexY, 0)*MyWeightsList[0][l].Segment);
				  }
				  else
				  {
					// Si es cero (menor que cero no existiŕia físicamente), la atenuación de todo el píxel es la del punto de entrada
					// del píxel:
					attenuationWeight = attenuationFactorEntryLimit;
				  }
				  attenuationFactorEntryLimit = attenuationFactorOutLimit;
				  
				  distanceValue += MyWeightsList[0][l].Segment;
				  // En attenuation value tengo la suma de los mu *largos de pixel, para covnertirlo en el 
				  // factor de atenuación tengo que hacer exp(-attenuationValue):
				  // Incremento el bin, con el valor de la proyección geométrica multiplicado por el factor de atenuación.  
				  if(attenuationWeight != 0)
					outputProjection->incrementSinogramBin(i,j, geometricValue * (1/distanceValue) * attenuationWeight);
				  
				} 
			  }
			  free(MyWeightsList[0]);
			}
		  }
		}
	  }
	}
  }
  return true;
  
}

/// Sobrecarga que realiza la Backprojection de EstimatedSinogram compensando por atenuación.
bool ConeOfResponseProjectorWithAttenuation::Backproject (Sinogram2Dtgs* inputSinogram, Image* outputImage)
{
  Point2D P1, P2;
  Point2D pointOnDetector;
  Line2D LOR;
  float rayLengthInFov_mm;
  float geometricValue, distanceValue;
  float attenuationLength, attenuationFactorEntryLimit, attenuationFactorOutLimit, attenuationWeight;
  float xPixel_mm, yPixel_mm;
  SiddonSegment** MyWeightsList = (SiddonSegment**)malloc(sizeof(SiddonSegment*));
  // Tamaño de la imagen:
  SizeImage sizeImage = outputImage->getSize();
  // Puntero a los píxeles:
  float* ptrPixels = outputImage->getPixelsPtr();
  
  // Para la corrección por atenuación utilizo el método de chang, que necesito la suma de todas las
  // distancias al detector, y dividir por la cantidad de lors para las cual fue calculado. Para esto
  // genero dos imágenes: una con la sumas de exp(-m*l) y en otra la cantidad de lors que la intersectaron:
  Image* sumAttenuationFactors = new Image(sizeImage);
  Image* numIntersectionsPerPixel = new Image(sizeImage);
  sumAttenuationFactors->fillConstant(0);
  numIntersectionsPerPixel->fillConstant(0);
  
  /// Para el cono de respuesta necesito obtener los distintos puntos sobre el dectector,
  /// y sobre la cara del colimador que me generan todas las LORs que generan el cono de respuesta.
  /// Para esto a partir del diámetro del colimador obtengo el paso entre cada punto, y el punto 
  /// inicial para luego ser usado en el for.
  float stepOnDetector_mm = inputSinogram->getWidthHoleCollimator_mm() /  numSamplesOnDetector;
  float firstPointOnDetector_mm = -(inputSinogram->getWidthHoleCollimator_mm()/2) + stepOnDetector_mm /2;
  float lastPointOnDetector_mm = (inputSinogram->getWidthHoleCollimator_mm()/2) - stepOnDetector_mm /2;
  
  for(int i = 0; i < inputSinogram->getNumProj(); i+=15)
  {
	for(int j = 0; j < inputSinogram->getNumR(); j++)
	{
	  for(float offsetDetector = firstPointOnDetector_mm; offsetDetector <= lastPointOnDetector_mm; offsetDetector+=stepOnDetector_mm)
	  {
		// Necesito la distancia del detector al primer píxel intersectado por siddon, para tenerlo en cuenta
		// en el factor del cuadrado de la distancia. Para esto necesito el punto sobre el detector:
		pointOnDetector = inputSinogram->getPointOnDetector(i, j, offsetDetector);
		for(float offsetCaraColimador = firstPointOnDetector_mm; offsetCaraColimador <= lastPointOnDetector_mm; offsetCaraColimador+=stepOnDetector_mm)
		{
		  inputSinogram->getPointsFromTgsLor(i, j, offsetDetector, offsetCaraColimador, &P1, &P2);
		  LOR.P0 = P1;
		  LOR.Vx = P2.X - P1.X;
		  LOR.Vy = P2.Y - P1.Y;
		  // Then I look for the intersection between the 3D LOR and the lines that
		  // delimits the voxels
		  // Siddon
		  
		  int LengthList;		  
		  geometricValue = 0;
		  // Para la implementación de la atenuación utilizo la implementación propuesta en: 
		  // Gullberg et al, An attenuated projector-backprojector for iterative SPECT reconstruction, Phys Med Biol, 1985, Vol 30, No 8
		  // No tiene ningñun truco en especial, es muy similar a lo que hacía yo originalmente, pero considera un promedio del coeficiente
		  // de atenuación en el píxel (a partir den una integral), en cambio yo usaba el del extremo.
		  attenuationLength = 0;	// En el paper variable ATENL.
		  attenuationFactorEntryLimit = exp(-attenuationLength);	// Peso de atenuación total en el punto de entrada al píxel a procesar
																// en el paper EX1.
		  attenuationFactorOutLimit = exp(-attenuationLength);	// Peso de atenuación total en el punto de salida al píxel a procesar
																// en el paper EX2.
		  attenuationWeight = 0;	// Peso de atenuación para una lor y píxel dado. Es el factor multiplicativo aplicado a la proyección
									// geométrica.
									
		  rayLengthInFov_mm = Siddon(LOR, outputImage, MyWeightsList, &LengthList,1);
		  if((LengthList!=0)&&(rayLengthInFov_mm!=0)) //Hay un caso que me está dano lengthlist=1 y rayLengthInFov_mm=0
		  {
			// Obtengo las coordenadas del primer píxel de entrada al fov.
			outputImage->getPixelGeomCoord(MyWeightsList[0][0].IndexX, MyWeightsList[0][0].IndexY, &xPixel_mm, &yPixel_mm);
			// Con esas coordenadas y el punto del detector puedo obtener la distancia del detector a la entrada
			// del fov, que va a ser el valor inicial del valor de distancia, que se usará para aplicar el factor
			// de ivnerso de la distancia:
			distanceValue = sqrt((pointOnDetector.X-xPixel_mm) * (pointOnDetector.X-xPixel_mm) +
			  (pointOnDetector.Y-yPixel_mm) * (pointOnDetector.Y-yPixel_mm));
			// Donde entro al FoV, supongo que todo lo anterior fue aire, por lo que la atenuación incial, es
			// la distancia a la entrada, por el coeficiente de atenuación del aire en 1/mm:
			attenuationLength = distanceValue * 0.0000097063f;
			attenuationFactorEntryLimit = exp(-attenuationLength);
			for(int l = 0; l < LengthList; l++)
			{
			  // for every element of the systema matrix different from zero,we do
			  // the sum(Aij*bi/Projected) for every i
			  if((MyWeightsList[0][l].IndexY>=0) && (MyWeightsList[0][l].IndexX>=0)&&(MyWeightsList[0][l].IndexY<sizeImage.nPixelsY) && (MyWeightsList[0][l].IndexX<sizeImage.nPixelsX))
			  {
				// El geomtricValue va incrementando valor de pixel * siddon.
				geometricValue = (MyWeightsList[0][l].Segment/rayLengthInFov_mm) * inputSinogram->getSinogramBin(i,j);
				distanceValue += MyWeightsList[0][l].Segment;
				
				// Para el atenuation value tengo que sumar todos los mu_pixel * siddon:
				attenuationLength += attenuationMap->getPixelValue(MyWeightsList[0][l].IndexX, MyWeightsList[0][l].IndexY, 0) * MyWeightsList[0][l].Segment; 
				attenuationFactorOutLimit = exp(-attenuationLength);
				
				// Para método de chang:
				sumAttenuationFactors->incrementPixelValue(MyWeightsList[0][l].IndexX, MyWeightsList[0][l].IndexY, 0,attenuationFactorOutLimit);
				numIntersectionsPerPixel->incrementPixelValue(MyWeightsList[0][l].IndexX, MyWeightsList[0][l].IndexY, 0,1);

				if((attenuationMap->getPixelValue(MyWeightsList[0][l].IndexX, MyWeightsList[0][l].IndexY, 0)>0)&&(MyWeightsList[0][l].Segment!=0)&&(attenuationFactorEntryLimit!=attenuationFactorOutLimit))
				{
				  // Si el mu de atenuación del píxel es mayor que cero, el factor de atenuación es igual a 
				  // (attenuationFactorEntryLimit-attenuationFactorOutLimit/mu_pixel. Esto sale de integrar la atenuación
				  // a lo largo del segmento que cruza el píxel:
				  attenuationWeight = (attenuationFactorEntryLimit-attenuationFactorOutLimit) / (attenuationMap->getPixelValue(MyWeightsList[0][l].IndexX, MyWeightsList[0][l].IndexY, 0)*MyWeightsList[0][l].Segment);
				}
				else
				{
				  // Si es cero (menor que cero no existiŕia físicamente), la atenuación de todo el píxel es la del punto de entrada
				  // del píxel:
				  attenuationWeight = attenuationFactorEntryLimit;
				}
				//printf("attenuationWeight: %f. attenuationLength: %f\n", attenuationWeight, attenuationLength);
				attenuationFactorEntryLimit = attenuationFactorOutLimit;
				if(geometricValue!=geometricValue)
				{
				  printf("Error.");
				}
				/*if(attenuationWeight==0)
				{
				  printf("AttenuationWeight 0. %d", l);
				}*/
				// En attenuation value tengo la suma de los mu *largos de pixel, para covnertirlo en el 
				// factor de atenuación tengo que hacer exp(-attenuationValue).
				// Actualizo la imagen:
				if((attenuationWeight!=0)&&(distanceValue!=0))
				{
				  ptrPixels[MyWeightsList[0][l].IndexY * sizeImage.nPixelsX + MyWeightsList[0][l].IndexX] += geometricValue; 
// 				  ptrPixels[MyWeightsList[0][l].IndexY * sizeImage.nPixelsX + MyWeightsList[0][l].IndexX] += geometricValue * (distanceValue); 
// 				  ptrPixels[MyWeightsList[0][l].IndexY * sizeImage.nPixelsX + MyWeightsList[0][l].IndexX] += geometricValue * distanceValue / attenuationWeight; 
// 				  ptrPixels[MyWeightsList[0][l].IndexY * sizeImage.nPixelsX + MyWeightsList[0][l].IndexX] += geometricValue / attenuationWeight;
				}
			  }
			}
			free(MyWeightsList[0]);
		  }
		}
	  }
	}
  }
  
  sumAttenuationFactors->writeInterfile("attFactors");
  //Normalizo el factor de corrección de chang y lo aplico:
  for(int i = 0; i < sizeImage.nPixelsX; i++)
  {
	for(int j = 0; j < sizeImage.nPixelsY; j++)
	{
	  if((numIntersectionsPerPixel->getPixelValue(i,j,0)!=0)&&(sumAttenuationFactors->getPixelValue(i,j,0)!=0))
	  {
		sumAttenuationFactors->setPixelValue(i,j,0,1/(sumAttenuationFactors->getPixelValue(i,j,0)/numIntersectionsPerPixel->getPixelValue(i,j,0)));
// 		sumAttenuationFactors->setPixelValue(i,j,0,1/(sumAttenuationFactors->getPixelValue(i,j,0)/(inputSinogram->getNumProj()*inputSinogram->getNumR())));
		outputImage->setPixelValue(i,j,0, outputImage->getPixelValue(i,j,0)*sumAttenuationFactors->getPixelValue(i,j,0));
	  }
	}
  }
  
  numIntersectionsPerPixel->writeInterfile("numIntersected");
  sumAttenuationFactors->writeInterfile("factors");
  
  return true;
}



