/**
	\file ConeOfResponseWithPenetrationProjectorWithAttenuation.cpp
	\brief Archivo que contiene la implementación de la clase ConeOfResponseProjector.

	\todo 
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2010.11.04
	\version 1.1.0
*/

#include <Siddon.h>
#include <ConeOfResponseWithPenetrationProjectorWithAttenuation.h>

ConeOfResponseWithPenetrationProjectorWithAttenuation::ConeOfResponseWithPenetrationProjectorWithAttenuation(int nSamplesOnDetector, int nSamplesOnCollimatorSurf, float linAttCoef_cm, Image* attImage)
{
  this->numSamplesOnDetector = nSamplesOnDetector;
  this->numSamplesOnCollimatorSurf = nSamplesOnCollimatorSurf;
  this->numLinesPerCone = nSamplesOnDetector * nSamplesOnDetector;
  this->linearAttenuationCoeficcient_cm = linAttCoef_cm;
  // La calculo a partir de la ecuación I=I0*exp(-mu*l). Como utilizo la atenuación at= (1-I/I0), y el coef
  // de atenuación lineal está en 1/cm, el máximo largo queda dado por log(1-at)/(-mu/10))
  this->lengthInCollimatorThreshold_mm = log(1-attenuationThreshold) / (-this->linearAttenuationCoeficcient_cm/10); 
  
  this->attenuationMap = new Image(attImage);
  // Creo un vector con todos los largos de segmentos para cada lor de entrada a un detector. Es independiente
  // del bin, solo tengo que tener en cuenta la cantidad de lors y los offsets con las que se forma cada una de
  // ellas. A REALIZAR
  
}

float ConeOfResponseWithPenetrationProjectorWithAttenuation::getAttenuationWeight(float lengthSegment_mm)
{
  // Neceito la relación I/I0, que va a ser el factor de atenuación, y es igual exp(-mu*l)
  float attWeight = exp(-this->linearAttenuationCoeficcient_cm * lengthSegment_mm / 10);
}

// Falta atenuación.
bool ConeOfResponseWithPenetrationProjectorWithAttenuation::Backproject (Sinogram2Dtgs* InputSinogram, Image* outputImage)
{
  Point2D P1, P2;
  Line2D LOR;
  Point2D pointOnDetector;
  float rayLengthInFov_mm, sumRayLengths_mm;
  float geometricValue, distanceValue;
  float xPixel_mm, yPixel_mm;
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
	  sumRayLengths_mm = 0;
	  // Peso total para esta LOR: sumo todas las distancias.
	  for(float offsetDetector = firstPointOnDetector_mm; offsetDetector <= lastPointOnDetector_mm; offsetDetector+=stepOnDetector_mm)
	  {
		// Necesito la distancia del detector al primer píxel intersectado por siddon, para tenerlo en cuenta
		// en el factor del cuadrado de la distancia. Para esto necesito el punto sobre el detector:
		pointOnDetector = InputSinogram->getPointOnDetector(i, j, offsetDetector);
		for(float offsetCaraColimador = firstPointOnDetector_mm; offsetCaraColimador <= lastPointOnDetector_mm; offsetCaraColimador+=stepOnDetector_mm)
		{
		  InputSinogram->getPointsFromTgsLor(i, j, offsetDetector, offsetCaraColimador, &P1, &P2);
		  LOR.P0 = P1;
		  LOR.Vx = P2.X - P1.X;
		  LOR.Vy = P2.Y - P1.Y;
		  rayLengthInFov_mm = getRayLengthInFov(LOR, outputImage);
		  sumRayLengths_mm += rayLengthInFov_mm;
		}
	  }
	  for(float offsetDetector = firstPointOnDetector_mm; offsetDetector <= lastPointOnDetector_mm; offsetDetector+=stepOnDetector_mm)
	  {
		// Necesito la distancia del detector al primer píxel intersectado por siddon, para tenerlo en cuenta
		// en el factor del cuadrado de la distancia. Para esto necesito el punto sobre el detector:
		pointOnDetector = InputSinogram->getPointOnDetector(i, j, offsetDetector);
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
			unsigned int LengthList;
			geometricValue = 0;
			attenuationWeight = getAttenuationWeight(lengthInCollimator);
			rayLengthInFov_mm = Siddon(LOR, outputImage, MyWeightsList, &LengthList,1);
			if(rayLengthInFov_mm > 0)
			{
			  // Obtengo las coordenadas del primer píxel de entrada al fov.
			  outputImage->getPixelGeomCoord(MyWeightsList[0][0].IndexX, MyWeightsList[0][0].IndexY, &xPixel_mm, &yPixel_mm);
			  // Con esas coordenadas y el punto del detector puedo obtener la distancia del detector a la entrada
			  // del fov, que va a ser el valor inicial del valor de distancia, que se usará para aplicar el factor
			  // de ivnerso de la distancia:
			  distanceValue = sqrt((pointOnDetector.X-xPixel_mm) * (pointOnDetector.X-xPixel_mm) +
				(pointOnDetector.Y-yPixel_mm) * (pointOnDetector.Y-yPixel_mm));
			  for(int l = 0; l < LengthList; l++)
			  {
				// for every element of the systema matrix different from zero,we do
				// the sum(Aij*bi/Projected) for every i
				if((MyWeightsList[0][l].IndexY>=0) && (MyWeightsList[0][l].IndexX>=0)&&(MyWeightsList[0][l].IndexY<sizeImage.nPixelsY) && (MyWeightsList[0][l].IndexX<sizeImage.nPixelsX))
				{
				  geometricValue = (MyWeightsList[0][l].Segment/sumRayLengths_mm) * InputSinogram->getSinogramBin(i,j);
				  //geometricValue = (MyWeightsList[0][l].Segment/rayLengthInFov_mm) * InputSinogram->getSinogramBin(i,j);
				  distanceValue += MyWeightsList[0][l].Segment;  
				  ptrPixels[MyWeightsList[0][l].IndexY * sizeImage.nPixelsX + MyWeightsList[0][l].IndexX] +=
					attenuationWeight * geometricValue;// * distanceValue;
				}
				
			  }
			  // Now I have my estimated projection for LOR i
			  if(LengthList!=0)	free(MyWeightsList[0]);
			}
		  }
		}
	  }
	}
  }
  return true;
}


bool ConeOfResponseWithPenetrationProjectorWithAttenuation::Project(Image* inputImage, Sinogram2Dtgs* outputProjection)
{
  Point2D P1, P2;
  Point2D pointOnDetector;
  Line2D LOR;
  float rayLengthInFov_mm;
  float geometricValue, distanceValue;
  float attenuationLength, attenuationFactorEntryLimit, attenuationFactorOutLimit, attenuationWeight;
  float xPixel_mm, yPixel_mm;
  float lengthInCollimator, attenuationWeightInCollimator;
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
  float margenDetector_mm = 12.4;
  float margenCollimador_mm = 35;
  float stepOnDetector_mm = (outputProjection->getWidthDetector_mm()-2*margenDetector_mm) /  numSamplesOnDetector;
  float stepOnCollimator_mm = (outputProjection->getWidthCollimator_mm()-2*margenCollimador_mm)  /  numSamplesOnCollimatorSurf;
  float firstPointOnDetector_mm = -(outputProjection->getWidthDetector_mm()/2-margenDetector_mm) + stepOnDetector_mm /2;
  float lastPointOnDetector_mm = (outputProjection->getWidthDetector_mm()/2-margenDetector_mm) - stepOnDetector_mm /2;
  float firstPointOnCollimatorSurf_mm = -(outputProjection->getWidthCollimator_mm()/2-margenCollimador_mm) + stepOnDetector_mm /2;
  float lastPointOnCollimatorSurf_mm = (outputProjection->getWidthCollimator_mm()/2-margenCollimador_mm) - stepOnDetector_mm /2;
  
  for(unsigned int i = 0; i < outputProjection->getNumProj(); i++)
  {
	
	for(unsigned int j = 0; j < outputProjection->getNumR(); j++)
	{
	  outputProjection->setSinogramBin(i,j,0);
	  for(float offsetDetector = firstPointOnDetector_mm; offsetDetector <= lastPointOnDetector_mm; offsetDetector+=stepOnDetector_mm)
	  {
		// Necesito la distancia del detector al primer píxel intersectado por siddon, para tenerlo en cuenta
		// en el factor del cuadrado de la distancia. Para esto necesito el punto sobre el detector:
		pointOnDetector = outputProjection->getPointOnDetector(i, j, offsetDetector);
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
			unsigned int LengthList;
			attenuationWeightInCollimator = getAttenuationWeight(lengthInCollimator);
			//printf("%f\n!", attenuationWeightInCollimator);
			geometricValue = 0;
			distanceValue = 0;
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
			rayLengthInFov_mm = Siddon(LOR, inputImage, MyWeightsList, &LengthList,1);
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
				attenuationLength = distanceValue * 0.0000097063;
				attenuationFactorEntryLimit = exp(-attenuationLength);
			
				for(unsigned int l = 0; l < LengthList; l++)
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
					outputProjection->incrementSinogramBin(i,j, attenuationWeight * geometricValue * (1/distanceValue) * attenuationWeightInCollimator);
				  } 
				
				}
				free(MyWeightsList[0]);	
			  }
			}
		  }
		}
	  }
	}
  }
  return true;
}

