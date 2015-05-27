/**
	\file SiddonProjectorWithAttenuation.cpp
	\brief Archivo que contiene la implementación de la clase SiddonProjector.

	\todo 
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2010.10.06
	\version 1.1.0
*/

#include <Siddon.h>
#include <SiddonProjectorWithAttenuation.h>

SiddonProjectorWithAttenuation::SiddonProjectorWithAttenuation(Image* attImage)
{
  this->numSamplesOnDetector = 1;  
  attenuationMap = new Image(attImage);
}

SiddonProjectorWithAttenuation::SiddonProjectorWithAttenuation(int nSamplesOnDetector, Image* attImage)
{
  this->numSamplesOnDetector = nSamplesOnDetector;  
  attenuationMap = new Image(attImage);
}


bool SiddonProjectorWithAttenuation::Project(Image* inputImage, Sinogram2Dtgs* outputProjection)
{
  Point2D P1, P2;
  Point2D pointOnDetector;
  Line2D LOR;
  float geometricValue, distanceValue;
  float attenuationLength, attenuationFactorEntryLimit, attenuationFactorOutLimit, attenuationWeight;
  float xPixel_mm, yPixel_mm;
  /// Después sacar:
  //SaveSystemMatrix(inputImage, outputProjection);
  
  SiddonSegment** MyWeightsList = (SiddonSegment**)malloc(sizeof(SiddonSegment*));
  // Tamaño de la imagen:
  SizeImage sizeImage = inputImage->getSize();
  // Puntero a los píxeles:
  float* ptrPixels = inputImage->getPixelsPtr();
  // Este proyector puede utilizar múltiples lors por bin del sinograma, pero siempre paralelas. 
  // Esto depende del parámetro numSamplesOnDetector. Las múltiples lors siempre están equiespaciadas:
  float stepOnDetector_mm = outputProjection->getWidthHoleCollimator_mm() /  numSamplesOnDetector;
  float firstPointOnDetector_mm = -(outputProjection->getWidthHoleCollimator_mm()/2) + stepOnDetector_mm /2;
  float lastPointOnDetector_mm = (outputProjection->getWidthHoleCollimator_mm()/2) - stepOnDetector_mm /2;
  
  // I only need to calculate the ForwardProjection of the bins were ara at least one event
  for(int i = 0; i < outputProjection->getNumProj(); i++)
  {
	for(int j = 0; j < outputProjection->getNumR(); j++)
	{
	  outputProjection->setSinogramBin(i,j,0);
	  for(float offsetDetector = firstPointOnDetector_mm; offsetDetector <= lastPointOnDetector_mm; offsetDetector+=stepOnDetector_mm)
	  {
		// Necesito la distancia del detector al primer píxel intersectado por siddon, para tenerlo en cuenta
		// en el factor del cuadrado de la distancia. Para esto necesito el punto sobre el detector:
		pointOnDetector = outputProjection->getPointOnDetector(i, j);
		outputProjection->getPointsFromTgsLor(i, j, offsetDetector, offsetDetector, &P1, &P2);
		LOR.P0 = P1;
		LOR.Vx = P2.X - P1.X;
		LOR.Vy = P2.Y - P1.Y;
		//printf("\n\n\nAngulo: %f R: %f. P1: %f %f. P2: %f %f.\n", outputProjection->getAngValue(i), outputProjection->getRValue(j), P1.X, P1.Y, P2.X, P2.Y);
		// Then I look for the intersection between the 3D LOR and the lines that
		// delimits the voxels
		// Siddon
		
		int LengthList;
		Siddon(LOR, inputImage, MyWeightsList, &LengthList,1);
		
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
								  
		distanceValue = 0;
		if(LengthList > 0)
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
			if((MyWeightsList[0][l].IndexY>=0) && (MyWeightsList[0][l].IndexX>=0))
			{
			  // El geomtricValue va incrementando valor de pixel * siddon.
			  geometricValue = MyWeightsList[0][l].Segment * 					 
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
			  //printf("Dist Value: %f. Attenuation Weight: %f. Pixel index: %d %d abs %d.\n", distanceValue, attenuationWeight, MyWeightsList[0][l].IndexX, MyWeightsList[0][l].IndexY, MyWeightsList[0][l].IndexY * sizeImage.nPixelsX + MyWeightsList[0][l].IndexX);
			  // Incremento el bin, con el valor de la proyección geométrica multiplicado por el factor de atenuación.  
			  outputProjection->incrementSinogramBin(i,j, geometricValue * (1/distanceValue) * attenuationWeight);
			  
			}
		  }
		  free(MyWeightsList[0]);
		}
		else
		{
		  printf("Lor sin pixels.");
		}
	  }
	}
  }
  return true;
}

bool SiddonProjectorWithAttenuation::Backproject(Sinogram2Dtgs* InputSinogram, Image* outputImage)
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
  // Este proyector puede utilizar múltiples lors por bin del sinograma, pero siempre paralelas. 
  // Esto depende del parámetro numSamplesOnDetector. Las múltiples lors siempre están equiespaciadas:
  float stepOnDetector_mm = InputSinogram->getWidthHoleCollimator_mm() /  numSamplesOnDetector;
  float firstPointOnDetector_mm = -(InputSinogram->getWidthHoleCollimator_mm()/2) + stepOnDetector_mm /2;
  float lastPointOnDetector_mm = (InputSinogram->getWidthHoleCollimator_mm()/2) - stepOnDetector_mm /2;
  
  // Para la corrección por atenuación utilizo el método de chang, que necesito la suma de todas las
  // distancias al detector, y dividir por la cantidad de lors para las cual fue calculado. Para esto
  // genero dos imágenes: una con la sumas de exp(-m*l) y en otra la cantidad de lors que la intersectaron:
  Image* sumAttenuationFactors = new Image(sizeImage);
  Image* numIntersectionsPerPixel = new Image(sizeImage);
  sumAttenuationFactors->fillConstant(0);
  numIntersectionsPerPixel->fillConstant(0);
  
  // I only need to calculate the ForwardProjection of the bins were ara at least one event
  for(int i = 0; i < InputSinogram->getNumProj(); i++)
  {
	for(int j = 0; j < InputSinogram->getNumR(); j++)
	{
	  for(float offsetDetector = firstPointOnDetector_mm; offsetDetector <= lastPointOnDetector_mm; offsetDetector+=stepOnDetector_mm)
	  {
		// Necesito la distancia del detector al primer píxel intersectado por siddon, para tenerlo en cuenta
		// en el factor del cuadrado de la distancia. Para esto necesito el punto sobre el detector:
		pointOnDetector = InputSinogram->getPointOnDetector(i, j);
		InputSinogram->getPointsFromTgsLor(i, j, offsetDetector, offsetDetector, &P1, &P2);
		LOR.P0 = P1;
		LOR.Vx = P2.X - P1.X;
		LOR.Vy = P2.Y - P1.Y;
		//printf("\n\n\nAngulo: %f R: %f. P1: %f %f. P2: %f %f.\n", InputSinogram->getAngValue(i), InputSinogram->getRValue(j), P1.X, P1.Y, P2.X, P2.Y);
		// Then I look for the intersection between the 3D LOR and the lines that
		// delimits the voxels
		// Siddon
		
		int LengthList;
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
		rayLengthInFov_mm = Siddon(LOR, outputImage, MyWeightsList, &LengthList,1);
		distanceValue = 0;
		if(rayLengthInFov_mm > 0)
		{
		  if(LengthList > 0)
		  {
			// Obtengo las coordenadas del primer píxel de entrada al fov.
			outputImage->getPixelGeomCoord(MyWeightsList[0][0].IndexX, MyWeightsList[0][0].IndexY, &xPixel_mm, &yPixel_mm);
			  
			// Con esas coordenadas y el punto del detector puedo obtener la distancia del detector a la entrada
			// del fov, que va a ser el valor inicial del valor de distancia, que se usará para aplicar el factor
			// de ivnerso de la distancia:
			distanceValue = sqrt((pointOnDetector.X-xPixel_mm) * (pointOnDetector.X-xPixel_mm) +
			  (pointOnDetector.Y-yPixel_mm) * (pointOnDetector.Y-yPixel_mm));
			//printf("\n\n\nAngulo: %f R: %f. P1: %f %f. P2: %f %f. Dist: %f\n", InputSinogram->getAngValue(i), InputSinogram->getRValue(j), P1.X, P1.Y, P2.X, P2.Y, distanceValue);
		
			// Donde entro al FoV, supongo que todo lo anterior fue aire, por lo que la atenuación incial, es
			// la distancia a la entrada, por el coeficiente de atenuación del aire en 1/mm:
			attenuationLength = distanceValue * 0.0000097063f;
			attenuationFactorEntryLimit = exp(-attenuationLength);
			for(int l = 0; l < LengthList; l++)
			{
			  if((MyWeightsList[0][l].IndexY>=0) && (MyWeightsList[0][l].IndexX>=0))
			  {
				// El geomtricValue va incrementando valor de pixel * siddon.
				geometricValue = MyWeightsList[0][l].Segment / sizeImage.sizePixelX_mm * InputSinogram->getSinogramBin(i,j);
				//geometricValue = MyWeightsList[0][l].Segment / rayLengthInFov_mm * InputSinogram->getSinogramBin(i,j);
				// Para el atenuation value tengo que sumar todos los mu_pixel * siddon:
				attenuationLength += attenuationMap->getPixelValue(MyWeightsList[0][l].IndexX, MyWeightsList[0][l].IndexY, 0) * MyWeightsList[0][l].Segment; 
				attenuationFactorOutLimit = exp(-attenuationLength);
				sumAttenuationFactors->incrementPixelValue(MyWeightsList[0][l].IndexX, MyWeightsList[0][l].IndexY, 0,attenuationFactorOutLimit);
				numIntersectionsPerPixel->incrementPixelValue(MyWeightsList[0][l].IndexX, MyWeightsList[0][l].IndexY, 0,1);
				if((attenuationMap->getPixelValue(MyWeightsList[0][l].IndexX, MyWeightsList[0][l].IndexY, 0)>0)&& (MyWeightsList[0][l].Segment!=0)&&(attenuationFactorEntryLimit!=attenuationFactorOutLimit))
				{
				  // Si el mu de atenuación del píxel es mayor que cero, el factor de atenuación es igual a 
				  // (attenuationFactorEntryLimit-attenuationFactorOutLimit/mu_pixel. Esto sale de integrar la atenuación
				  // a lo largo del segmento que cruza el píxel:
				  //attenuationWeight = (attenuationFactorEntryLimit+attenuationFactorOutLimit)/2;
				  attenuationWeight = (attenuationFactorEntryLimit-attenuationFactorOutLimit) / (attenuationMap->getPixelValue(MyWeightsList[0][l].IndexX, MyWeightsList[0][l].IndexY, 0)*(MyWeightsList[0][l].Segment / sizeImage.sizePixelX_mm));
				}
				else
				{
				  // Si es cero (menor que cero no existiŕia físicamente), la atenuación de todo el píxel es la del punto de entrada
				  // del píxel:
				  attenuationWeight = attenuationFactorEntryLimit;
				}
				attenuationFactorEntryLimit = attenuationFactorOutLimit;
					  
				distanceValue += MyWeightsList[0][l].Segment;
				//printf("Dist Value: %f. Attenuation Weight: %f. Pixel index: %d %d abs %d.\n", distanceValue, attenuationWeight, MyWeightsList[0][l].IndexX, MyWeightsList[0][l].IndexY, MyWeightsList[0][l].IndexY * sizeImage.nPixelsX + MyWeightsList[0][l].IndexX);
				// En attenuation value tengo la suma de los mu *largos de pixel, para covnertirlo en el 
				// factor de atenuación tengo que hacer exp(-attenuationValue).
				// Actualizo la imagen:
				if((attenuationWeight!=0))
				{
				  ptrPixels[MyWeightsList[0][l].IndexY * sizeImage.nPixelsX + MyWeightsList[0][l].IndexX] += geometricValue;	
// 				  ptrPixels[MyWeightsList[0][l].IndexY * sizeImage.nPixelsX + MyWeightsList[0][l].IndexX] += geometricValue * (1/attenuationWeight) * distanceValue;	
// 				  ptrPixels[MyWeightsList[0][l].IndexY * sizeImage.nPixelsX + MyWeightsList[0][l].IndexX] += geometricValue;
				  //if((i==15)||(i==45))
					//printf("Dist Value: %f. Attenuation Weight: %f. Pixel index: %d %d abs %d.\n", distanceValue, attenuationWeight, MyWeightsList[0][l].IndexX, MyWeightsList[0][l].IndexY, MyWeightsList[0][l].IndexY * sizeImage.nPixelsX + MyWeightsList[0][l].IndexX);
				
				}
			  }
			}
			// Now I have my estimated projection for LOR i
			free(MyWeightsList[0]);
		  }
		}
	  }
	}
  }
  sumAttenuationFactors->writeInterfile("attFactors");
  // Normalizo el factor de corrección de chang y lo aplico:
  for(int i = 0; i < sizeImage.nPixelsX; i++)
  {
	for(int j = 0; j < sizeImage.nPixelsY; j++)
	{
	  if((numIntersectionsPerPixel->getPixelValue(i,j,0)!=0)&&(sumAttenuationFactors->getPixelValue(i,j,0)!=0))
	  {
		sumAttenuationFactors->setPixelValue(i,j,0,1/(sumAttenuationFactors->getPixelValue(i,j,0)/numIntersectionsPerPixel->getPixelValue(i,j,0)));
		outputImage->setPixelValue(i,j,0, outputImage->getPixelValue(i,j,0)*sumAttenuationFactors->getPixelValue(i,j,0));
	  }
	}
  }
  
  numIntersectionsPerPixel->writeInterfile("numIntersected");
  sumAttenuationFactors->writeInterfile("factors");
  return true;
}

/** Sección para Sinogram3D. */
bool SiddonProjectorWithAttenuation::Project (Image* inputImage, Sinogram3D* outputProjection)
{
	return false;
}
// 	Point3D P1, P2;
// 	Line3D LOR;
// 	SiddonSegment** MyWeightsList = (SiddonSegment**)malloc(sizeof(SiddonSegment*));
// 	// Tamaño de la imagen:
// 	SizeImage sizeImage = inputImage->getSize();
// 	// Puntero a los píxeles:
// 	float* ptrPixels = inputImage->getPixelsPtr();
// 	// Inicializo el sino 3D con el que voy a trabajar en cero.
// 	// Lo incializo acá porque despues distintas cobinaciones de z aportan al mismo bin.
// 	outputProjection->FillConstant(0.0);
// 	for(unsigned int i = 0; i < outputProjection->CantSegmentos; i++)
// 	{
// 		printf("Forwardprojection Segmento: %d\n", i);
// 		for(unsigned int j = 0; j < outputProjection->Segmentos[i]->CantSinogramas; j++)
// 		{
// 			printf("\t\tSinograma: %d\n", j);
// 			/// Cálculo de las coordenadas z del sinograma
// 			for(unsigned int m = 0; m < outputProjection->Segmentos[i]->Sinogramas2D[j]->getNumZ(); m++)
// 			{
// 			  int indexRing1 = outputProjection->Segmentos[i]->Sinogramas2D[j]->getRing1FromList(m);
// 			  int indexRing2 = outputProjection->Segmentos[i]->Sinogramas2D[j]->getRing2FromList(m);
// 			  
// 			  for(unsigned int k = 0; k < outputProjection->Segmentos[i]->Sinogramas2D[j]->getNumProj(); k++)
// 			  {
// 			    for(unsigned int l = 0; l < outputProjection->Segmentos[i]->Sinogramas2D[j]->getNumR(); l++)
// 			    {
// 			      /// Cada Sinograma 2D me represnta múltiples LORs, según la mínima y máxima diferencia entre anillos.
// 			      /// Por lo que cada bin me va a sumar cuentas en lors con distintos ejes axiales.
// 			      /// El sinograma de salida lo incializo en cero antes de recorrer los distintos anillos de cada elemento del
// 			      /// sinograma, ya que varias LORS deben aportar al mismo bin del sinograma.
// 
// 			      GetPointsFromLOR(outputProjection->Segmentos[i]->Sinogramas2D[j]->getAngValue(k), outputProjection->Segmentos[i]->Sinogramas2D[j]->getRValue(l), 
// 					      outputProjection->ZValues[indexRing1], outputProjection->ZValues[indexRing2], outputProjection->Rscanner, &P1, &P2);
// 			      LOR.P0 = P1;
// 			      LOR.Vx = P2.X - P1.X;
// 			      LOR.Vy = P2.Y - P1.Y;
// 			      LOR.Vz = P2.Z - P1.Z;
// 			      // Then I look for the intersection between the 3D LOR and the lines that
// 			      // delimits the voxels
// 			      // Siddon					
// 			      unsigned int LengthList;
// 			      Siddon(LOR, inputImage, MyWeightsList, &LengthList,1);
// 			      
// 			      if(LengthList > 0)
// 			      {
// 				      /// Hay elementos dentro del FOV
// 				      for(unsigned int n = 0; n < LengthList; n++)
// 				      {
// 					      // for every element of the systema matrix different from zero,we do
// 					      // the sum(Aij*Xj) for every J
// 					      if((MyWeightsList[0][n].IndexX<sizeImage.nPixelsX)&&(MyWeightsList[0][n].IndexY<sizeImage.nPixelsY)&&(MyWeightsList[0][n].IndexZ<sizeImage.nPixelsZ))
// 					      {		
// 						      outputProjection->Segmentos[i]->Sinogramas2D[j]->incrementSinogramBin(k, l, MyWeightsList[0][n].Segment * 
// 							      ptrPixels[MyWeightsList[0][n].IndexZ*(sizeImage.nPixelsX*sizeImage.nPixelsY)+MyWeightsList[0][n].IndexY * sizeImage.nPixelsX + MyWeightsList[0][n].IndexX]);
// 					      }
// 					      
// 					      /*else
// 					      {
// 						      printf("Siddon fuera\n");
// 					      }*/
// 					      //printf("r:%d phi:%d z1:%d z2:%d x:%d y:%d z:%d w:%f", j, i, indexRing1, indexRing2, MyWeightsList[0][l].IndexX, MyWeightsList[0][l].IndexY, MyWeightsList[0][l].IndexZ, MyWeightsList[0][l].Segment);	
// 				      }
// 				      if(LengthList != 0)
// 				      {
// 					      free(MyWeightsList[0]);
// 				      }
// 				      
// 			      }
// 			    }
// 					// Now I have my estimated projection for LOR i
// 			  }
// 			}
// 		}
// 	}
// 
// 	return true;
// }

bool SiddonProjectorWithAttenuation::Backproject (Sinogram3D* inputProjection, Image* outputImage)
{
	return false;
}
// 	Point3D P1, P2;
// 	Line3D LOR;
// 	SiddonSegment** MyWeightsList = (SiddonSegment**)malloc(sizeof(SiddonSegment*));
// 	// Tamaño de la imagen:
// 	SizeImage sizeImage = outputImage->getSize();
// 	// Puntero a los píxeles:
// 	float* ptrPixels = outputImage->getPixelsPtr();
// 	for(unsigned int i = 0; i < inputProjection->CantSegmentos; i++)
// 	{
// 		printf("Backprojection Segmento: %d\n", i);
// 		for(unsigned int j = 0; j < inputProjection->Segmentos[i]->CantSinogramas; j++)
// 		{
// 			/// Cálculo de las coordenadas z del sinograma
// 			printf("\t\tSinograma: %d\n", j);
// 			for(unsigned int k = 0; k < inputProjection->Segmentos[i]->Sinogramas2D[j]->getNumProj(); k++)
// 			{
// 				for(unsigned int l = 0; l < inputProjection->Segmentos[i]->Sinogramas2D[j]->getNumR(); l++)
// 				{
// 					/// Cada Sinograma 2D me represnta múltiples LORs, según la mínima y máxima diferencia entre anillos.
// 					/// Por lo que cada bin me va a sumar cuentas en lors con distintos ejes axiales.
// 					if(inputProjection->Segmentos[i]->Sinogramas2D[j]->getSinogramBin(k,l) != 0)
// 					{
// 						for(unsigned int m = 0; m < inputProjection->Segmentos[i]->Sinogramas2D[j]->getNumZ(); m++)
// 						{
// 							int indexRing1 = inputProjection->Segmentos[i]->Sinogramas2D[j]->getRing1FromList(m);
// 							int indexRing2 = inputProjection->Segmentos[i]->Sinogramas2D[j]->getRing2FromList(m);
// 							GetPointsFromLOR(inputProjection->Segmentos[i]->Sinogramas2D[j]->getAngValue(k), inputProjection->Segmentos[i]->Sinogramas2D[j]->getRValue(l), 
// 									inputProjection->ZValues[indexRing1], inputProjection->ZValues[indexRing2], inputProjection->Rscanner, &P1, &P2);
// 							LOR.P0 = P1;
// 							LOR.Vx = P2.X - P1.X;
// 							LOR.Vy = P2.Y - P1.Y;
// 							LOR.Vz = P2.Z - P1.Z;
// 							// Then I look for the intersection between the 3D LOR and the lines that
// 							// delimits the voxels
// 							// Siddon					
// 							unsigned int LengthList;
// 							Siddon(LOR, outputImage, MyWeightsList, &LengthList,1);
// 							for(unsigned int n = 0; n < LengthList; n++)
// 							{
// 								// for every element of the systema matrix different from zero,we do
// 								// the sum(Aij*bi/Projected) for every i
// 								if((MyWeightsList[0][n].IndexZ<sizeImage.nPixelsZ)&&(MyWeightsList[0][n].IndexY<sizeImage.nPixelsY)&&(MyWeightsList[0][n].IndexX<sizeImage.nPixelsX))
// 								{
// 									ptrPixels[MyWeightsList[0][n].IndexZ*(sizeImage.nPixelsX*sizeImage.nPixelsY)+MyWeightsList[0][n].IndexY * sizeImage.nPixelsX + MyWeightsList[0][n].IndexX] +=
// 										MyWeightsList[0][n].Segment * inputProjection->Segmentos[i]->Sinogramas2D[j]->getSinogramBin(k,l);	
// 								}
// 								/*else
// 								{
// 									printf("Siddon fuera de mapa\n");
// 								}*/
// 							}
// 							if(LengthList != 0)
// 							{
// 							  /// Solo libero memoria cuando se la pidió, si no hay una excepción.
// 							  free(MyWeightsList[0]);
// 							}
// 						}
// 					}
// 					// Now I have my estimated projection for LOR i
// 				}
// 			}
// 		}
// 	}
// 
// 	return true;
// }


bool SiddonProjectorWithAttenuation::SaveSystemMatrix(Image* inputImage, Sinogram2Dtgs* outputProjection)
{
  Point2D P1, P2;
  Point2D pointOnDetector;
  Line2D LOR;
  float geometricValue, distanceValue;
  float attenuationLength, attenuationFactorEntryLimit, attenuationFactorOutLimit, attenuationWeight;
  float xPixel_mm, yPixel_mm;
  float* systemMatrix;
  
  SiddonSegment** MyWeightsList = (SiddonSegment**)malloc(sizeof(SiddonSegment*));
  // Tamaño de la imagen:
  SizeImage sizeImage = inputImage->getSize();
  // Puntero a los píxeles:
  float* ptrPixels = inputImage->getPixelsPtr();
  int colsSmr = sizeImage.nPixelsX*sizeImage.nPixelsY;
  int rowsSmr = outputProjection->getNumProj()* outputProjection->getNumR();
  systemMatrix = (float*)malloc(rowsSmr*colsSmr);
  // Este proyector puede utilizar múltiples lors por bin del sinograma, pero siempre paralelas. 
  // Esto depende del parámetro numSamplesOnDetector. Las múltiples lors siempre están equiespaciadas:
  float stepOnDetector_mm = outputProjection->getWidthHoleCollimator_mm() /  numSamplesOnDetector;
  float firstPointOnDetector_mm = -(outputProjection->getWidthHoleCollimator_mm()/2) + stepOnDetector_mm /2;
  float lastPointOnDetector_mm = (outputProjection->getWidthHoleCollimator_mm()/2) - stepOnDetector_mm /2;
  
  // I only need to calculate the ForwardProjection of the bins were ara at least one event
  for(int i = 0; i < outputProjection->getNumProj(); i++)
  {
	for(int j = 0; j < outputProjection->getNumR(); j++)
	{
	  for(float offsetDetector = firstPointOnDetector_mm; offsetDetector <= lastPointOnDetector_mm; offsetDetector+=stepOnDetector_mm)
	  {
		// Necesito la distancia del detector al primer píxel intersectado por siddon, para tenerlo en cuenta
		// en el factor del cuadrado de la distancia. Para esto necesito el punto sobre el detector:
		pointOnDetector = outputProjection->getPointOnDetector(i, j);
		outputProjection->getPointsFromTgsLor(i, j, offsetDetector, offsetDetector, &P1, &P2);
		LOR.P0 = P1;
		LOR.Vx = P2.X - P1.X;
		LOR.Vy = P2.Y - P1.Y;
		//printf("\n\n\nAngulo: %f R: %f. P1: %f %f. P2: %f %f.\n", outputProjection->getAngValue(i), outputProjection->getRValue(j), P1.X, P1.Y, P2.X, P2.Y);
		// Then I look for the intersection between the 3D LOR and the lines that
		// delimits the voxels
		// Siddon
		
		int LengthList;
		Siddon(LOR, inputImage, MyWeightsList, &LengthList,1);
		outputProjection->setSinogramBin(i,j,0);
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
								  
		distanceValue = 0;
		if(LengthList > 0)
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
			if((MyWeightsList[0][l].IndexY>=0) && (MyWeightsList[0][l].IndexX>=0))
			{
			  // El geomtricValue va incrementando valor de pixel * siddon.
			  geometricValue = MyWeightsList[0][l].Segment * 					 
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
			  //printf("Dist Value: %f. Attenuation Weight: %f. Pixel index: %d %d abs %d.\n", distanceValue, attenuationWeight, MyWeightsList[0][l].IndexX, MyWeightsList[0][l].IndexY, MyWeightsList[0][l].IndexY * sizeImage.nPixelsX + MyWeightsList[0][l].IndexX);
			  // Incremento el bin, con el valor de la proyección geométrica multiplicado por el factor de atenuación.  
			  outputProjection->incrementSinogramBin(i,j, geometricValue * (1/distanceValue) * attenuationWeight);
			  int row = (i*outputProjection->getNumR())+j;
			  int col = MyWeightsList[0][l].IndexY * sizeImage.nPixelsX + MyWeightsList[0][l].IndexX;
			  systemMatrix[row*colsSmr+col] = geometricValue * (1/distanceValue) * attenuationWeight;
			}
		  }
		  free(MyWeightsList[0]);
		}
		else
		{
		  printf("Lor sin pixels.");
		}
	  }
	}
  }
  FILE* fileSmr = fopen("SiddonwA_SMR.dat","wb");
  const unsigned int SizeData = rowsSmr*colsSmr;
  if((fwrite(systemMatrix, sizeof(float), SizeData, fileSmr)) !=  SizeData)
	return false;
  fclose(fileSmr);
  return true;
  return true;
}