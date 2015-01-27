/**
	\file SiddonProjector.cpp
	\brief Archivo que contiene la implementación de la clase SiddonProjector.

	\todo 
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2010.10.06
	\version 1.1.0
*/

#include <Siddon.h>
#include <SiddonProjector.h>

SiddonProjector::SiddonProjector()
{
  this->numSamplesOnDetector = 1;  
}

SiddonProjector::SiddonProjector(int nSamplesOnDetector)
{
  this->numSamplesOnDetector = nSamplesOnDetector;  
}


bool SiddonProjector::Backproject (Sinogram2D* InputSinogram, Image* outputImage)
{
  Point2D P1, P2;
  Line2D LOR;
  SiddonSegment* MyWeightsList;
  // Tamaño de la imagen:
  SizeImage sizeImage = outputImage->getSize();
  // Puntero a los píxeles:
  float* ptrPixels = outputImage->getPixelsPtr();
  unsigned int LengthList, i, j, l, indexPixel;
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
	      // for every element of the systema matrix different from zero,we do
	      // the sum(Aij*bi/Projected) for every i
	      if((MyWeightsList[l].IndexY>=0) && (MyWeightsList[l].IndexX>=0) && 
		(MyWeightsList[l].IndexY<sizeImage.nPixelsY) && (MyWeightsList[l].IndexX<sizeImage.nPixelsX))
	      {
		indexPixel = MyWeightsList[l].IndexY * sizeImage.nPixelsX + MyWeightsList[l].IndexX;
		newValue = MyWeightsList[l].Segment * InputSinogram->getSinogramBin(i,j)*geomFactor;
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
bool SiddonProjector::DivideAndBackproject (Sinogram2D* InputSinogram, Sinogram2D* EstimatedSinogram, Image* outputImage)
{
  Point2D P1, P2;
  Line2D LOR;
  SiddonSegment* MyWeightsList;
  // Tamaño de la imagen:
  SizeImage sizeImage = outputImage->getSize();
  // Puntero a los píxeles:
  float* ptrPixels = outputImage->getPixelsPtr();
  unsigned int LengthList, i, j, l, indexPixel;
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
		// for every element of the systema matrix different from zero,we do
		// the sum(Aij*bi/Projected) for every i
		if((MyWeightsList[l].IndexY>=0) && (MyWeightsList[l].IndexX>=0) && 
		  (MyWeightsList[l].IndexY<sizeImage.nPixelsY) && (MyWeightsList[l].IndexX<sizeImage.nPixelsX))
		{
		  indexPixel = MyWeightsList[l].IndexY * sizeImage.nPixelsX + MyWeightsList[l].IndexX;
		    newValue = geomFactor * MyWeightsList[l].Segment * InputSinogram->getSinogramBin(i,j) / EstimatedSinogram->getSinogramBin(i,j);
		    
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

bool SiddonProjector::Project (Image* inputImage, Sinogram2D* outputProjection)
{
  Point2D P1, P2;
  Line2D LOR;
  SiddonSegment** MyWeightsList;
  // Tamaño de la imagen:
  SizeImage sizeImage = inputImage->getSize();
  // Puntero a los píxeles:
  float* ptrPixels = inputImage->getPixelsPtr();
  unsigned int LengthList, i, j, l;
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
	      if((MyWeightsList[0][l].IndexY>=0) && (MyWeightsList[0][l].IndexX>=0))
		outputProjection->incrementSinogramBin(i,j, geomFactor * MyWeightsList[0][l].Segment * 
		  ptrPixels[MyWeightsList[0][l].IndexY * sizeImage.nPixelsX + MyWeightsList[0][l].IndexX]);		  
	    }
	    if(outputProjection->getSinogramBin(i,j) != outputProjection->getSinogramBin(i,j))
	      printf("Warnign: NaN.\n");
	    free(MyWeightsList[0]);
	  }
	}
      }
    }
  }
  return true;
}


bool SiddonProjector::Backproject (Sinogram2Dtgs* InputSinogram, Image* outputImage)
{
  Point2D P1, P2;
  Line2D LOR;
  SiddonSegment** MyWeightsList = (SiddonSegment**)malloc(sizeof(SiddonSegment*));
  // Tamaño de la imagen:
  SizeImage sizeImage = outputImage->getSize();
  // Puntero a los píxeles:
  float* ptrPixels = outputImage->getPixelsPtr();
  // I only need to calculate the ForwardProjection of the bins were ara at least one event
  for(int i = 0; i < InputSinogram->getNumProj(); i++)
  {
    for(int j = 0; j < InputSinogram->getNumR(); j++)
    {
      GetPointsFromTgsLor (InputSinogram->getAngValue(i), InputSinogram->getRValue(j), InputSinogram->getDistCrystalToCenterFov(), InputSinogram->getLengthColimator_mm(), &P1, &P2);
      LOR.P0 = P1;
      LOR.Vx = P2.X - P1.X;
      LOR.Vy = P2.Y - P1.Y;
      // Then I look for the intersection between the 3D LOR and the lines that
      // delimits the voxels
      // Siddon
      
      unsigned int LengthList;
      Siddon(LOR, outputImage, MyWeightsList, &LengthList,1);
      for(int l = 0; l < LengthList; l++)
      {
	  // for every element of the systema matrix different from zero,we do
	  // the sum(Aij*bi/Projected) for every i
	  if((MyWeightsList[0][l].IndexY>=0) && (MyWeightsList[0][l].IndexX>=0))
		ptrPixels[MyWeightsList[0][l].IndexY * sizeImage.nPixelsX + MyWeightsList[0][l].IndexX] +=
		  MyWeightsList[0][l].Segment * InputSinogram->getSinogramBin(i,j) / InputSinogram->getSinogramBin(i, j);				
      }
      // Now I have my estimated projection for LOR i
      free(MyWeightsList[0]);
    }
  }
  return true;
}

/// Sobrecarga que realiza la Backprojection de InputSinogram/EstimatedSinogram
bool SiddonProjector::DivideAndBackproject (Sinogram2Dtgs* InputSinogram, Sinogram2Dtgs* EstimatedSinogram, Image* outputImage)
{
	Point2D P1, P2;
	Line2D LOR;
	SiddonSegment** MyWeightsList = (SiddonSegment**)malloc(sizeof(SiddonSegment*));
	// Tamaño de la imagen:
	SizeImage sizeImage = outputImage->getSize();
	// Puntero a los píxeles:
	float* ptrPixels = outputImage->getPixelsPtr();
	// I only need to calculate the ForwardProjection of the bins were ara at least one event
	for(int i = 0; i < InputSinogram->getNumProj(); i++)
	{
		for(int j = 0; j < InputSinogram->getNumR(); j++)
		{
		  GetPointsFromTgsLor (InputSinogram->getAngValue(i), InputSinogram->getRValue(j), InputSinogram->getDistCrystalToCenterFov(), InputSinogram->getLengthColimator_mm(), &P1, &P2);
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
			  if((MyWeightsList[0][l].IndexY>=0) && (MyWeightsList[0][l].IndexX>=0))
				ptrPixels[MyWeightsList[0][l].IndexY * sizeImage.nPixelsX + MyWeightsList[0][l].IndexX] +=
				  MyWeightsList[0][l].Segment * InputSinogram->getSinogramBin(i,j) / EstimatedSinogram->getSinogramBin(i,j);	
		  }
		  // Now I have my estimated projection for LOR i
		  free(MyWeightsList[0]);
		}
	}
	return true;
}

bool SiddonProjector::Project (Image* inputImage, Sinogram2Dtgs* outputProjection)
{
	Point2D P1, P2;
	Line2D LOR;
	SiddonSegment** MyWeightsList = (SiddonSegment**)malloc(sizeof(SiddonSegment*));
	// Tamaño de la imagen:
	SizeImage sizeImage = inputImage->getSize();
	// Puntero a los píxeles:
	float* ptrPixels = inputImage->getPixelsPtr();
	// I only need to calculate the ForwardProjection of the bins were ara at least one event
	for(unsigned int i = 0; i < outputProjection->getNumProj(); i++)
	{
		for(unsigned int j = 0; j < outputProjection->getNumR(); j++)
		{
		  // The bin is different zero so we start calculatin forward project
		  // First we get the values of the SystemMAtrix for this LOR
		  // First I get the line parameters for the 3D LOR
		  // First I get the line parameters for the 3D LOR
		  /// Modificación para el TGS:
		  GetPointsFromTgsLor (outputProjection->getAngValue(i), outputProjection->getRValue(j), outputProjection->getDistCrystalToCenterFov(), outputProjection->getLengthColimator_mm(), &P1, &P2);
		  LOR.P0 = P1;
		  LOR.Vx = P2.X - P1.X;
		  LOR.Vy = P2.Y - P1.Y;
		  // Then I look for the intersection between the 3D LOR and the lines that
		  // delimits the voxels
		  // Siddon
		  
		  unsigned int LengthList;
		  Siddon(LOR, inputImage, MyWeightsList, &LengthList,1);
		  outputProjection->setSinogramBin(i,j,0);
		  for(unsigned int l = 0; l < LengthList; l++)
		  {
			  // for every element of the systema matrix different from zero,we do
			  // the sum(Aij*Xj) for every J
			  if((MyWeightsList[0][l].IndexY>=0) && (MyWeightsList[0][l].IndexX>=0))
				outputProjection->incrementSinogramBin(i,j, MyWeightsList[0][l].Segment * 
				  ptrPixels[MyWeightsList[0][l].IndexY * sizeImage.nPixelsX + MyWeightsList[0][l].IndexX]);  
			  //printf("r:%d phi:%d z1:%d z2:%d x:%d y:%d z:%d w:%f", j, i, indexRing1, indexRing2, MyWeightsList[0][l].IndexX, MyWeightsList[0][l].IndexY, MyWeightsList[0][l].IndexZ, MyWeightsList[0][l].Segment);	
		  
		  }
		  free(MyWeightsList[0]);
		}
	}
	return true;
}

/** Sección para Sinogram3D. */
bool SiddonProjector::Project (Image* inputImage, Sinogram3D* outputProjection)
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
  for(unsigned int i = 0; i < outputProjection->getNumSegments(); i++)
  {
    printf("Forwardprojection con Siddon Segmento: %d\n", i);	  
    for(unsigned int j = 0; j < outputProjection->getSegment(i)->getNumSinograms(); j++)
    {
      /// Cálculo de las coordenadas z del sinograma
      for(unsigned int m = 0; m < outputProjection->getSegment(i)->getSinogram2D(j)->getNumZ(); m++)
      {
	int indexRing1 = outputProjection->getSegment(i)->getSinogram2D(j)->getRing1FromList(m);
	int indexRing2 = outputProjection->getSegment(i)->getSinogram2D(j)->getRing2FromList(m);
	unsigned int k, l, n, LengthList;
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
		      // El axial depende del ancho del ring
		      // Por ahora deshabilito el GeomFactor:
// 		      outputProjection->getSegment(i)->getSinogram2D(j)->incrementSinogramBin(k, l, MyWeightsList[0][n].Segment * geomFactor *
// 			    ptrPixels[MyWeightsList[0][n].IndexZ*(sizeImage.nPixelsX*sizeImage.nPixelsY)+MyWeightsList[0][n].IndexY * sizeImage.nPixelsX + MyWeightsList[0][n].IndexX]);
		      outputProjection->getSegment(i)->getSinogram2D(j)->incrementSinogramBin(k, l, MyWeightsList[0][n].Segment *
			    ptrPixels[MyWeightsList[0][n].IndexZ*(sizeImage.nPixelsX*sizeImage.nPixelsY)+MyWeightsList[0][n].IndexY * sizeImage.nPixelsX + MyWeightsList[0][n].IndexX]);
		    }	
		    
	    
		  }
		   
		  if(LengthList != 0)
		  {
		    free(MyWeightsList[0]);
		  }
		}
		else
		{
		  printf("No hay lista de intersección. LOR Segmento: %d. Sino %d. R: %d. Theta: %d.\n", i, j, k, l);
		    
		}
// 		if(outputProjection->getSegment(i)->getSinogram2D(j)->getSinogramBin(k,l)==0)
// 		{
// 		  printf("Cero en el bin Segmento: %d. Sino %d. R: %d. Theta: %d.\n", i, j, k, l);
// 		}
// 		if(j==12)
// 		      printf("value:%f.", outputProjection->getSegment(i)->getSinogram2D(j)->getSinogramBin(k,l));
	      }
	    }
		    // Now I have my estimated projection for LOR i
	  }
	}
	
// 	printf("Bin de ejemplo: %d %d %f.\n", i,j,outputProjection->getSegment(i)->getSinogram2D(j)->getSinogramBin(20,20));
      }
    }
  }
  return true;
}


bool SiddonProjector::Backproject (Sinogram3D* inputProjection, Image* outputImage)
{
  Point3D P1, P2;
  Line3D LOR;
  SiddonSegment** MyWeightsList;
  // Tamaño de la imagen:
  SizeImage sizeImage = outputImage->getSize();
  // Puntero a los píxeles:
  float* ptrPixels = outputImage->getPixelsPtr();
  unsigned int i, j, k, l, n, m, LengthList;
  unsigned long indexPixel;
  float geomFactor = 0;
  float newValue;
  #pragma omp parallel private(i, j, k, l, m, LOR, P1, P2, MyWeightsList, LengthList, n, newValue, indexPixel, geomFactor) shared(inputProjection,ptrPixels)
  {
    MyWeightsList = (SiddonSegment**)malloc(sizeof(SiddonSegment*));
    for(i = 0; i < inputProjection->getNumSegments(); i++)
    {
      printf("Backprojection con Siddon Segmento: %d\n", i);
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
		      
		      indexPixel = MyWeightsList[0][n].IndexZ*(sizeImage.nPixelsX*sizeImage.nPixelsY)+MyWeightsList[0][n].IndexY * sizeImage.nPixelsX + MyWeightsList[0][n].IndexX;
		      // Por ahora deshabilito el GeomFactor:
// 		      newValue = MyWeightsList[0][n].Segment * geomFactor * inputProjection->getSegment(i)->getSinogram2D(j)->getSinogramBin(k,l);
		      newValue = MyWeightsList[0][n].Segment * inputProjection->getSegment(i)->getSinogram2D(j)->getSinogramBin(k,l);
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
bool SiddonProjector::DivideAndBackproject (Sinogram3D* InputSinogram3D, Sinogram3D* EstimatedSinogram3D, Image* outputImage)
{
  Point3D P1, P2;
  Line3D LOR;
  SiddonSegment** MyWeightsList;
  // Tamaño de la imagen:
  SizeImage sizeImage = outputImage->getSize();
  // Puntero a los píxeles:
  float* ptrPixels = outputImage->getPixelsPtr();
  unsigned int i, j, k, l, n, m, LengthList;
  unsigned long indexPixel;
  float newValue;
  float geomFactor = 0;
  #pragma omp parallel private(i, j, k, l, m, LOR, P1, P2, MyWeightsList, LengthList, n, newValue, indexPixel, geomFactor) shared(InputSinogram3D,EstimatedSinogram3D,ptrPixels)
  {
    MyWeightsList = (SiddonSegment**)malloc(sizeof(SiddonSegment*));
    for(i = 0; i < InputSinogram3D->getNumSegments(); i++)
    {
      printf("Backprojection con Siddon Segmento: %d\n", i);
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
		      // Por ahora deshabilito el GeomFactor:
		      /*if(EstimatedSinogram3D->getSegment(i)->getSinogram2D(j)->getSinogramBin(k,l)!=0)
			newValue = MyWeightsList[0][n].Segment * geomFactor * InputSinogram3D->getSegment(i)->getSinogram2D(j)->getSinogramBin(k,l) /
			  EstimatedSinogram3D->getSegment(i)->getSinogram2D(j)->getSinogramBin(k,l);*/	
		      if(EstimatedSinogram3D->getSegment(i)->getSinogram2D(j)->getSinogramBin(k,l)!=0)
			newValue = MyWeightsList[0][n].Segment * InputSinogram3D->getSegment(i)->getSinogram2D(j)->getSinogramBin(k,l) /
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