/**
	\file ArPetProjector.cpp
	\brief Archivo que contiene la implementación de la clase ArPetProjector. Es similar al proyector de Siddon pero tiene en cuenta la geometría del AR-PET, calculando el ángulo
	sólido de emisión.

	\todo 
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2010.10.06
	\version 1.1.0
*/

#include <Siddon.h>
#include <ArPetProjector.h>

ArPetProjector::ArPetProjector()
{
  this->numSamplesOnDetector = 1;  
}

ArPetProjector::ArPetProjector(int nSamplesOnDetector)
{
  this->numSamplesOnDetector = nSamplesOnDetector;  
}


bool ArPetProjector::Backproject (Sinogram2D* InputSinogram, Image* outputImage)
{
  Point2D P1, P2;
  Line2D LOR;
  SiddonSegment* MyWeightsList;
  // Tamaño de la imagen:
  SizeImage sizeImage = outputImage->getSize();
  // Puntero a los píxeles:
  float* ptrPixels = outputImage->getPixelsPtr();
  int LengthList, i, j, l, indexPixel;
  float newValue = 0, geomFactor;
  // I only need to calculate the ForwardProjection of the bins were ara at least one event
  #pragma omp parallel private(i, j, l, LOR, P1, P2, MyWeightsList, geomFactor, LengthList, newValue, indexPixel) shared(InputSinogram,ptrPixels)
  {
    //MyWeightsList = (SiddonSegment**)malloc(sizeof(SiddonSegment*));
    #pragma omp for
    for(int i = 0; i < InputSinogram->getNumProj(); i++)
    {
      for(int j = 0; j < InputSinogram->getNumR(); j++)
      {
	// Obtengo puntos de entrada y salida de la lor:
	if(InputSinogram->getPointsFromLor(i, j, &P1, &P2, &geomFactor))
	{
	  LOR.P0 = P1;
	  LOR.Vx = P2.X - P1.X;
	  LOR.Vy = P2.Y - P1.Y;
	  //Siddon(P1, P2, outputImage, &MyWeightsList, &LengthList,1);
	  float rayLength = Siddon (LOR, outputImage, &MyWeightsList, &LengthList,1);
	  if(LengthList>0)
	  {
	    for(l = 0; l < LengthList; l++)
	    {
	      float x_mm, y_mm, z_mm;
	      outputImage->getPixelGeomCoord(MyWeightsList[l].IndexX,MyWeightsList[l].IndexY,MyWeightsList[l].IndexZ,
		&x_mm, &y_mm, &z_mm);
	      // Calculo distancia a cada uno de los puntos:
	      float distXY1 = sqrt((P1.X-x_mm)*(P1.X-x_mm)+(P1.Y-y_mm)*(P1.Y-y_mm));
	      float distXY2 = sqrt((P2.X-x_mm)*(P2.X-x_mm)+(P2.Y-y_mm)*(P2.Y-y_mm));

	      // Necesito los dos ángulos que forman el ángulo sólido, el transversal y el axial. El primero depende de la mayor distancia al detector en el plano xy:
	      if (distXY2 > distXY1)
		distXY1 = distXY2;

	      if((MyWeightsList[l].IndexY>=0) && (MyWeightsList[l].IndexX>=0) && 
		(MyWeightsList[l].IndexY<sizeImage.nPixelsY) && (MyWeightsList[l].IndexX<sizeImage.nPixelsX))
	      {
		indexPixel = MyWeightsList[l].IndexY * sizeImage.nPixelsX + MyWeightsList[l].IndexX;
		newValue = MyWeightsList[l].Segment * InputSinogram->getSinogramBin(i,j)*geomFactor*1/(distXY1);
		#pragma omp atomic
		  ptrPixels[indexPixel] += newValue;	
	      }
	    }
	    // Now I have my estimated projection for LOR i
	    free(MyWeightsList);
	  }
	}
      }
    }
  }
  return true;
}

/// Sobrecarga que realiza la Backprojection de InputSinogram/EstimatedSinogram
bool ArPetProjector::DivideAndBackproject (Sinogram2D* InputSinogram, Sinogram2D* EstimatedSinogram, Image* outputImage)
{
  Point2D P1, P2;
  Line2D LOR;
  SiddonSegment* MyWeightsList;
  // Tamaño de la imagen:
  SizeImage sizeImage = outputImage->getSize();
  // Puntero a los píxeles:
  float* ptrPixels = outputImage->getPixelsPtr();
  int LengthList, i, j, l, indexPixel;
  float newValue = 0, geomFactor;
  // I only need to calculate the ForwardProjection of the bins were ara at least one event
  #pragma omp parallel private(i, j, l, LOR, P1, P2, MyWeightsList, geomFactor, LengthList, newValue, indexPixel) shared(InputSinogram,EstimatedSinogram,ptrPixels)
  {
    //MyWeightsList = (SiddonSegment**)malloc(sizeof(SiddonSegment*));
    #pragma omp for
    for(i = 0; i < InputSinogram->getNumProj(); i++)
    {
      for(j = 0; j < InputSinogram->getNumR(); j++)
      {
	if((EstimatedSinogram->getSinogramBin(i,j)!=0)&&(InputSinogram->getSinogramBin(i,j)!=0))
	{
	  if(InputSinogram->getPointsFromLor(i, j, &P1, &P2, &geomFactor))
	  {
	    LOR.P0 = P1;
	    LOR.Vx = P2.X - P1.X;
	    LOR.Vy = P2.Y - P1.Y;
	    //Siddon(P1, P2, outputImage, &MyWeightsList, &LengthList,1);
	    float rayLength = Siddon (LOR, outputImage, &MyWeightsList, &LengthList,1);
	    if(LengthList>0)
	    {
	      for(l = 0; l < LengthList; l++)
	      {
		float x_mm, y_mm, z_mm;
		outputImage->getPixelGeomCoord(MyWeightsList[l].IndexX,MyWeightsList[l].IndexY,MyWeightsList[l].IndexZ,
		  &x_mm, &y_mm, &z_mm);
		// Calculo distancia a cada uno de los puntos:
		float distXY1 = sqrt((P1.X-x_mm)*(P1.X-x_mm)+(P1.Y-y_mm)*(P1.Y-y_mm));
		float distXY2 = sqrt((P2.X-x_mm)*(P2.X-x_mm)+(P2.Y-y_mm)*(P2.Y-y_mm));

		// Necesito los dos ángulos que forman el ángulo sólido, el transversal y el axial. El primero depende de la mayor distancia al detector en el plano xy:
		if (distXY2 > distXY1)
		  distXY1 = distXY2;
		if((MyWeightsList[l].IndexY>=0) && (MyWeightsList[l].IndexX>=0) && 
		  (MyWeightsList[l].IndexY<sizeImage.nPixelsY) && (MyWeightsList[l].IndexX<sizeImage.nPixelsX))
		{
		  indexPixel = MyWeightsList[l].IndexY * sizeImage.nPixelsX + MyWeightsList[l].IndexX;
		    newValue = geomFactor * MyWeightsList[l].Segment * InputSinogram->getSinogramBin(i,j) / EstimatedSinogram->getSinogramBin(i,j)*1/(distXY1);
		    
		  #pragma omp atomic
		    ptrPixels[indexPixel] += newValue;	
		}
	      }
	      // Now I have my estimated projection for LOR i
	      if(MyWeightsList!=NULL)
		free(MyWeightsList);
	    }
	  }
	}
      }
    }
  }
  return true;
}

bool ArPetProjector::Project (Image* inputImage, Sinogram2D* outputProjection)
{
  Point2D P1, P2;
  Line2D LOR;
  SiddonSegment** MyWeightsList;
  // Tamaño de la imagen:
  SizeImage sizeImage = inputImage->getSize();
  // Puntero a los píxeles:
  float* ptrPixels = inputImage->getPixelsPtr();
  int LengthList, i, j, l;
  float geomFactor;
  // I only need to calculate the ForwardProjection of the bins were ara at least one event
  #pragma omp parallel private(i, j, l, LOR, P1, P2, MyWeightsList, geomFactor, LengthList) shared(outputProjection,ptrPixels)
  {
    MyWeightsList = (SiddonSegment**)malloc(sizeof(SiddonSegment*));
    #pragma omp for
    for(int i = 0; i < outputProjection->getNumProj(); i++)
    {
      for(int j = 0; j < outputProjection->getNumR(); j++)
      {
	if(outputProjection->getPointsFromLor(i, j, &P1, &P2, &geomFactor))
	{
	  LOR.P0 = P1;
	  LOR.Vx = P2.X - P1.X;
	  LOR.Vy = P2.Y - P1.Y;
	  //Siddon(P1, P2, inputImage, MyWeightsList, &LengthList,1); // Este se usa cuando los puntos son los límites del fov.
	  float rayLength = Siddon (LOR, inputImage, MyWeightsList, &LengthList,1);
	  outputProjection->setSinogramBin(i,j,0);
	  if(LengthList > 0)
	  {
	    for(l = 0; l < LengthList; l++)
	    {
	      float x_mm, y_mm, z_mm;
	      inputImage->getPixelGeomCoord(MyWeightsList[0][l].IndexX,MyWeightsList[0][l].IndexY,MyWeightsList[0][l].IndexZ,
		&x_mm, &y_mm, &z_mm);
	      // Calculo distancia a cada uno de los puntos:
	      float distXY1 = sqrt((P1.X-x_mm)*(P1.X-x_mm)+(P1.Y-y_mm)*(P1.Y-y_mm));
	      float distXY2 = sqrt((P2.X-x_mm)*(P2.X-x_mm)+(P2.Y-y_mm)*(P2.Y-y_mm));

	      // Necesito los dos ángulos que forman el ángulo sólido, el transversal y el axial. El primero depende de la mayor distancia al detector en el plano xy:
	      if (distXY2 > distXY1)
		distXY1 = distXY2;
	      if((MyWeightsList[0][l].IndexY>=0) && (MyWeightsList[0][l].IndexX>=0))
		outputProjection->incrementSinogramBin(i,j, geomFactor * MyWeightsList[0][l].Segment *1/(distXY1)* 
		  ptrPixels[MyWeightsList[0][l].IndexY * sizeImage.nPixelsX + MyWeightsList[0][l].IndexX]);		  
	    }
	    if(outputProjection->getSinogramBin(i,j) != outputProjection->getSinogramBin(i,j))
	      printf("aca");
	    free(MyWeightsList[0]);
	  }
	}
      }
    }
  }
  return true;
}

/** Sección para Sinogram3D. */
bool ArPetProjector::Project (Image* inputImage, Sinogram3D* outputProjection)
{
  Point3D P1, P2;
  Line3D LOR;
  SiddonSegment** MyWeightsList;
  // Tamaño de la imagen:
  SizeImage sizeImage = inputImage->getSize();
  // Puntero a los píxeles:
  float* ptrPixels = inputImage->getPixelsPtr();
  float geomFactor = 0;
  // Inicializo el sino 3D con el que voy a trabajar en cero.
  // Lo incializo acá porque despues distintas cobinaciones de z aportan al mismo bin.
  outputProjection->FillConstant(0.0);
  outputProjection->writeInterfile("test");
  for(unsigned int i = 0; i < outputProjection->getNumSegments(); i++)
  {
    printf("Forwardprojection  con ArPetProjector Segmento: %d\n", i);
    for(unsigned int j = 0; j < outputProjection->getSegment(i)->getNumSinograms(); j++)
    {
      /// Cálculo de las coordenadas z del sinograma
      for(unsigned int m = 0; m < outputProjection->getSegment(i)->getSinogram2D(j)->getNumZ(); m++)
      {
	int indexRing1 = outputProjection->getSegment(i)->getSinogram2D(j)->getRing1FromList(m);
	int indexRing2 = outputProjection->getSegment(i)->getSinogram2D(j)->getRing2FromList(m);
	int k, l, n, LengthList;
	#pragma omp parallel private(k, l, LOR, MyWeightsList, LengthList, n, P1, P2, geomFactor)
	{
	  MyWeightsList = (SiddonSegment**)malloc(sizeof(SiddonSegment*));
	  #pragma omp for
	  for(k = 0; k < outputProjection->getSegment(i)->getSinogram2D(j)->getNumProj(); k++)
	  {
	    for(l = 0; l < outputProjection->getSegment(i)->getSinogram2D(j)->getNumR(); l++)
	    {
	      /// Cada Sinograma 2D me represnta múltiples LORs, según la mínima y máxima diferencia entre anillos.
	      /// Por lo que cada bin me va a sumar cuentas en lors con distintos ejes axiales.
	      /// El sinograma de salida lo incializo en cero antes de recorrer los distintos anillos de cada elemento del
	      /// sinograma, ya que varias LORS deben aportar al mismo bin del sinograma.
	      if(outputProjection->getSegment(i)->getSinogram2D(j)->getPointsFromLor(k,l,m, &P1, &P2, &geomFactor))
	      {
		LOR.P0 = P1;
		LOR.Vx = P2.X - P1.X;
		LOR.Vy = P2.Y - P1.Y;
		LOR.Vz = P2.Z - P1.Z;
		// Then I look for the intersection between the 3D LOR and the lines that
		// delimits the voxels
		// Siddon		
		Siddon(LOR, inputImage, MyWeightsList, &LengthList,1);
		
		if(LengthList > 0)
		{
		  /// Hay elementos dentro del FOV
		  for(n = 0; n < LengthList; n++)
		  {
		    // for every element of the systema matrix different from zero,we do
		    // the sum(Aij*Xj) for every J
		    if((MyWeightsList[0][n].IndexX<sizeImage.nPixelsX)&&(MyWeightsList[0][n].IndexY<sizeImage.nPixelsY)&&(MyWeightsList[0][n].IndexZ<sizeImage.nPixelsZ))
		    {
		      float x_mm, y_mm, z_mm;
		      inputImage->getPixelGeomCoord(MyWeightsList[0][n].IndexX,MyWeightsList[0][n].IndexY,MyWeightsList[0][n].IndexZ,
			&x_mm, &y_mm, &z_mm);
		      // Calculo distancia a cada uno de los puntos:
		      float distXY1 = sqrt((P1.X-x_mm)*(P1.X-x_mm)+(P1.Y-y_mm)*(P1.Y-y_mm));
		      float distXY2 = sqrt((P2.X-x_mm)*(P2.X-x_mm)+(P2.Y-y_mm)*(P2.Y-y_mm));
		      float distXY = distXY1+distXY2;
		      float dist1 = sqrt((P1.X-x_mm)*(P1.X-x_mm)+(P1.Z-z_mm)*(P1.Z-z_mm)+(P1.Y-y_mm)*(P1.Y-y_mm));
		      float dist2 = sqrt((P2.X-x_mm)*(P2.X-x_mm)+(P2.Z-z_mm)*(P2.Z-z_mm)+(P2.Y-y_mm)*(P2.Y-y_mm));//1/(dist1*dist1) * 1/(dist2*dist2) * 
		      float dist = dist1+dist2;

		      // Necesito los dos ángulos que forman el ángulo sólido, el transversal y el axial. El primero depende de la mayor distancia al detector en el plano xy:
		      if (distXY2 > distXY1)
			distXY1 = distXY2;
		      if (dist2 > dist1)
			dist1 = dist2;

		      // El axial depende del ancho del ring
		      outputProjection->getSegment(i)->getSinogram2D(j)->incrementSinogramBin(k, l, MyWeightsList[0][n].Segment * geomFactor * 1/(distXY1)* distXY/(dist)* distXY1/(dist1)*
			    ptrPixels[MyWeightsList[0][n].IndexZ*(sizeImage.nPixelsX*sizeImage.nPixelsY)+MyWeightsList[0][n].IndexY * sizeImage.nPixelsX + MyWeightsList[0][n].IndexX]);
		    }		    
		  }
		  if(LengthList != 0)
		  {
		    free(MyWeightsList[0]);
		  }
		}
	      }
	      /*else
	      {
		printf("LOR sin deteccion.\n");
	      }*/
	    }
		    // Now I have my estimated projection for LOR i
	  }
	}
      }
    }
  }
  return true;
}


bool ArPetProjector::Backproject (Sinogram3D* inputProjection, Image* outputImage)
{
  Point3D P1, P2;
  Line3D LOR;
  SiddonSegment** MyWeightsList;
  // Tamaño de la imagen:
  SizeImage sizeImage = outputImage->getSize();
  // Puntero a los píxeles:
  float* ptrPixels = outputImage->getPixelsPtr();
  int i, j, k, l, n, m, LengthList;
  unsigned long indexPixel;
  float geomFactor = 0;
  float newValue;
  #pragma omp parallel private(i, j, k, l, m, LOR, P1, P2, MyWeightsList, LengthList, n, newValue, indexPixel, geomFactor) shared(inputProjection,ptrPixels)
  {
    MyWeightsList = (SiddonSegment**)malloc(sizeof(SiddonSegment*));
    for(i = 0; i < inputProjection->getNumSegments(); i++)
    {
      printf("Backprojection  con ArPetProjector Segmento: %d\n", i);
      #pragma omp for
      for(j = 0; j < inputProjection->getSegment(i)->getNumSinograms(); j++)
      {
	/// Cálculo de las coordenadas z del sinograma
	for(k = 0; k < inputProjection->getSegment(i)->getSinogram2D(j)->getNumProj(); k++)
	{
	      
	  for(l = 0; l < inputProjection->getSegment(i)->getSinogram2D(j)->getNumR(); l++)
	  {
	    /// Cada Sinograma 2D me represnta múltiples LORs, según la mínima y máxima diferencia entre anillos.
	    /// Por lo que cada bin me va a sumar cuentas en lors con distintos ejes axiales.
	    if(inputProjection->getSegment(i)->getSinogram2D(j)->getSinogramBin(k,l) != 0)
	    {
	      for(m = 0; m < inputProjection->getSegment(i)->getSinogram2D(j)->getNumZ(); m++)
	      {
		int indexRing1 = inputProjection->getSegment(i)->getSinogram2D(j)->getRing1FromList(m);
		int indexRing2 = inputProjection->getSegment(i)->getSinogram2D(j)->getRing2FromList(m);
		// Devuelve true si encontró los dos cabezales de la LOR:
		if(inputProjection->getSegment(i)->getSinogram2D(j)->getPointsFromLor(k,l,m, &P1, &P2, &geomFactor))
		{
		  LOR.P0 = P1;
		  LOR.Vx = P2.X - P1.X;
		  LOR.Vy = P2.Y - P1.Y;
		  LOR.Vz = P2.Z - P1.Z;
		  
		  // Then I look for the intersection between the 3D LOR and the lines that
		  // delimits the voxels
		  // Siddon					
		  Siddon(LOR, outputImage, MyWeightsList, &LengthList,1);
		      
		  for(n = 0; n < LengthList; n++)
		  {
		    // for every element of the systema matrix different from zero,we do
		    // the sum(Aij*bi/Projected) for every i
		    if((MyWeightsList[0][n].IndexZ<sizeImage.nPixelsZ)&&(MyWeightsList[0][n].IndexY<sizeImage.nPixelsY)&&(MyWeightsList[0][n].IndexX<sizeImage.nPixelsX))
		    {
		      float x_mm, y_mm, z_mm;
		      outputImage->getPixelGeomCoord(MyWeightsList[0][n].IndexX,MyWeightsList[0][n].IndexY,MyWeightsList[0][n].IndexZ,
			&x_mm, &y_mm, &z_mm);
		      // Calculo distancia a cada uno de los puntos:
		      float distXY1 = sqrt((P1.X-x_mm)*(P1.X-x_mm)+(P1.Y-y_mm)*(P1.Y-y_mm));
		      float distXY2 = sqrt((P2.X-x_mm)*(P2.X-x_mm)+(P2.Y-y_mm)*(P2.Y-y_mm));
		      float distXY = distXY1+distXY2;
		      float dist1 = sqrt((P1.X-x_mm)*(P1.X-x_mm)+(P1.Z-z_mm)*(P1.Z-z_mm)+(P1.Y-y_mm)*(P1.Y-y_mm));
		      float dist2 = sqrt((P2.X-x_mm)*(P2.X-x_mm)+(P2.Z-z_mm)*(P2.Z-z_mm)+(P2.Y-y_mm)*(P2.Y-y_mm));//1/(dist1*dist1) * 1/(dist2*dist2) * 
		      float dist = dist1+dist2;

		      // Necesito los dos ángulos que forman el ángulo sólido, el transversal y el axial. El primero depende de la mayor distancia al detector en el plano xy:
		      if (distXY2 > distXY1)
			distXY1 = distXY2;
		      if (dist2 > dist1)
			dist1 = dist2;
		      indexPixel = MyWeightsList[0][n].IndexZ*(sizeImage.nPixelsX*sizeImage.nPixelsY)+MyWeightsList[0][n].IndexY * sizeImage.nPixelsX + MyWeightsList[0][n].IndexX;
		      newValue = MyWeightsList[0][n].Segment * geomFactor * 1/(distXY1)* distXY/(dist)* distXY1/(dist1) * inputProjection->getSegment(i)->getSinogram2D(j)->getSinogramBin(k,l);
		      #pragma omp atomic 
			ptrPixels[indexPixel] +=  newValue;	
		      
		    }
		    /*else
		    {
			    printf("Siddon fuera de mapa\n");
		    }*/
		  }
		  if(LengthList != 0)
		  {
		    /// Solo libero memoria cuando se la pidió, si no hay una excepción.
		    free(MyWeightsList[0]);
		  }
		}
	      }
	    }
		      // Now I have my estimated projection for LOR i
	  }
	}
      }
    }
  }
  return true;
}

/// Sobrecarga que realiza la Backprojection del cociente InputSinogram3D/EstimatedSinogram3D
bool ArPetProjector::DivideAndBackproject (Sinogram3D* InputSinogram3D, Sinogram3D* EstimatedSinogram3D, Image* outputImage)
{
  Point3D P1, P2;
  Line3D LOR;
  SiddonSegment** MyWeightsList;
  // Tamaño de la imagen:
  SizeImage sizeImage = outputImage->getSize();
  // Puntero a los píxeles:
  float* ptrPixels = outputImage->getPixelsPtr();
  int i, j, k, l, n, m, LengthList;
  unsigned long indexPixel;
  float newValue;
  float geomFactor = 0;
  #pragma omp parallel private(i, j, k, l, m, LOR, P1, P2, MyWeightsList, LengthList, n, newValue, indexPixel, geomFactor) shared(InputSinogram3D,EstimatedSinogram3D,ptrPixels)
  {
    MyWeightsList = (SiddonSegment**)malloc(sizeof(SiddonSegment*));
    for(i = 0; i < InputSinogram3D->getNumSegments(); i++)
    {
      printf("Backprojection  con ArPetProjector Segmento: %d\n", i);
      #pragma omp for
      for(j = 0; j < InputSinogram3D->getSegment(i)->getNumSinograms(); j++)
      {
	/// Cálculo de las coordenadas z del sinograma
	for(k = 0; k < InputSinogram3D->getSegment(i)->getSinogram2D(j)->getNumProj(); k++)
	{
	  for(l = 0; l < InputSinogram3D->getSegment(i)->getSinogram2D(j)->getNumR(); l++)
	  {
	    /// Cada Sinograma 2D me represnta múltiples LORs, según la mínima y máxima diferencia entre anillos.
	    /// Por lo que cada bin me va a sumar cuentas en lors con distintos ejes axiales.
	    if(InputSinogram3D->getSegment(i)->getSinogram2D(j)->getSinogramBin(k,l) != 0)
	    {
	      for(m = 0; m < InputSinogram3D->getSegment(i)->getSinogram2D(j)->getNumZ(); m++)
	      {
		int indexRing1 = InputSinogram3D->getSegment(i)->getSinogram2D(j)->getRing1FromList(m);
		int indexRing2 = InputSinogram3D->getSegment(i)->getSinogram2D(j)->getRing2FromList(m);
		if(InputSinogram3D->getSegment(i)->getSinogram2D(j)->getPointsFromLor(k,l,m, &P1, &P2, &geomFactor))
		{
		  LOR.P0 = P1;
		  LOR.Vx = P2.X - P1.X;
		  LOR.Vy = P2.Y - P1.Y;
		  LOR.Vz = P2.Z - P1.Z;
		  // Then I look for the intersection between the 3D LOR and the lines that
		  // delimits the voxels
		  // Siddon		
		  Siddon(LOR, outputImage, MyWeightsList, &LengthList,1);
		  for(n = 0; n < LengthList; n++)
		  {
		    // for every element of the systema matrix different from zero,we do
		    // the sum(Aij*bi/Projected) for every i
		    if((MyWeightsList[0][n].IndexZ<sizeImage.nPixelsZ)&&(MyWeightsList[0][n].IndexY<sizeImage.nPixelsY)&&(MyWeightsList[0][n].IndexX<sizeImage.nPixelsX))
		    {
		      indexPixel = MyWeightsList[0][n].IndexZ*(sizeImage.nPixelsX*sizeImage.nPixelsY)+MyWeightsList[0][n].IndexY * sizeImage.nPixelsX + MyWeightsList[0][n].IndexX;
		      if(EstimatedSinogram3D->getSegment(i)->getSinogram2D(j)->getSinogramBin(k,l)!=0)
			newValue = MyWeightsList[0][n].Segment * geomFactor * InputSinogram3D->getSegment(i)->getSinogram2D(j)->getSinogramBin(k,l) /
			  EstimatedSinogram3D->getSegment(i)->getSinogram2D(j)->getSinogramBin(k,l);	
		      else if(InputSinogram3D->getSegment(i)->getSinogram2D(j)->getSinogramBin(k,l)!=0)
		      {
			/// Estimated = 0, pero Input != 0. Mantengo el valor de Input.
			newValue = MyWeightsList[0][n].Segment * geomFactor * InputSinogram3D->getSegment(i)->getSinogram2D(j)->getSinogramBin(k,l);	
		      }
		      else
		      {
		      /// Los bins de los sinogramas Input y Estimates son 0, o sea tengo el valor indeterminado 0/0.
		      /// Lo más lógico pareciera ser dejarlo en 0 al cociente, para que no sume al backprojection.
		      /// Sumarle 0 es lo mismo que nada.
			newValue = 0;
		      }
		      
		      #pragma omp atomic 
			ptrPixels[indexPixel] +=  newValue;
		    }
		/*else
		{
			printf("Siddon fuera de mapa\n");
		}*/
		  }
		  if(LengthList != 0)
		  {
		    free(MyWeightsList[0]);
		  }
		}
	      }
	    }
	    // Now I have my estimated projection for LOR i
	  }
	}
      }
    }
  }
  return true;
}