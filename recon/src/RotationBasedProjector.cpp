/**
	\file RotationBasedProjector.cpp
	\brief Archivo que contiene la implementación de la clase RotationBasedProjector.

	\todo 
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2010.10.06
	\version 1.1.0
*/

#include <RotationBasedProjector.h>

RotationBasedProjector::RotationBasedProjector(InterpolationMethods intMethod)
{
  interpolationMethod = intMethod;
}

RotationBasedProjector::RotationBasedProjector(string intMethod)
{
	if(!strcmp(intMethod.c_str(),"nearest"))
		interpolationMethod = RotationBasedProjector::NEAREST;
	else if(!strcmp(intMethod.c_str(),"bilinear"))
		interpolationMethod = RotationBasedProjector::BILINEAR;
	else if(!strcmp(intMethod.c_str(),"bicubic"))
		interpolationMethod = RotationBasedProjector::BICUBIC;
	else
		// By default nearest:
		interpolationMethod = RotationBasedProjector::NEAREST;
}

bool RotationBasedProjector::RotateImage(Image* inputImage, Image* rotatedImage, float rotAngle_deg, InterpolationMethods interpMethod)
{
  float rotAngle_rad = rotAngle_deg * DEG_TO_RAD;
  // La imagen de salida, puede ser de distinto tamaño que la de entrada, por lo que además
  // de la rotación tengo que hacer un cambio de escala. Obtengo dichos factores de escala:
  SizeImage sizeInputImage = inputImage->getSize();
  SizeImage sizeRotatedImage = rotatedImage->getSize();
  // Los factores de escala los calculo de forma inversa: Entrada/Salida porque así aplico la transformación:
  float invScaleX = float(sizeInputImage.nPixelsX) / float(sizeRotatedImage.nPixelsX);
  float invScaleY = float(sizeInputImage.nPixelsX) / float(sizeRotatedImage.nPixelsY);
  // Rotación de la imagen.
  // Para hacer una transformación espacial de una imagen, siempre es recomendado hacerlo
  // de destino a fuente, para poder aplicar la interpolación deseada. El tamaño de una imagen
  // rotada cambia. La única forma que se mantenga el tamaño es rotando un FOV cricular, que es
  // el caso del pet. Entonces para este caso voy a rotar un fov circular. Para implementar esto
  // puedo recorrer los píxeles del fov circular, o calcular la rotación de todas las imágenes y
  // no actualizar los píxeles que se van de la misma. A su vez la rotación la hago desde el centro
  // de la imagen.

  // La coordenada de cada píxel es el centro del mismo. O sea la coordenada (0,0) es
  // el centro del píxel de arriba a la izquierda. Por lo que los límites de la imagen van
  // a estar en -0.5 y (nPixelsX-1+0.5) en cada eje. El centro de la imagen en índices de píxeles
  // es (sizeImage.nPixelsX-1)/2.
  float centroInputX_pixels = (float)(sizeInputImage.nPixelsX-1) / 2.0f;
  float centroInputY_pixels = (float)(sizeInputImage.nPixelsY-1) / 2.0f;
  float centroRotatedX_pixels = (float)(sizeRotatedImage.nPixelsX-1) / 2.0f;
  float centroRotatedY_pixels = (float)(sizeRotatedImage.nPixelsY-1) / 2.0f;
  float i_entrada, j_entrada, i_entrada_ent, j_entrada_ent, i_entrada_frac, j_entrada_frac, value;
  float radioFovCuadrado = (float)sizeRotatedImage.nPixelsX*sizeRotatedImage.nPixelsX/4.0f;
  // Podría poner los if para el método de interpolación dentro del for, porque la rotación
  // de coordenadas las hacen todas por igual, pero como la interpolación cambia bastante, separo
  // toda la operación directamente aunque se repita un poco de código, así no repite el if para
  // cada píxel (aunque es muy probable que el compilador lo haga automáticamente a eso).
  if(interpMethod == NEAREST)
  {
	for(int i = 0; i < sizeRotatedImage.nPixelsX; i++)
	{
	  for(int j = 0; j < sizeRotatedImage.nPixelsY; j++)
	  {
		//if((((float)i)-centroRotatedX_pixels)*((float)i-centroRotatedX_pixels)+(centroRotatedY_pixels-(float)j)*(centroRotatedY_pixels-(float)j) <= radioFovCuadrado)
		if(fovImage->getPixelValue(i,j,0) != 0)
		{
		  rotatedImage->setPixelValue(i, j, 0, 0);
		  // Primero la rotación, y luego el escalamiento (recordar que a partir del píxel de salida, quiero obtener
		  // el de entrada por lo que la matriz de rotación y de escalamiento es la inversa de la transformación geométrica
		  // deseada:
		  i_entrada = (i-centroRotatedX_pixels) * cos(rotAngle_rad) + (centroRotatedY_pixels-j) * sin(rotAngle_rad);
		  j_entrada = (-(i-centroRotatedX_pixels) * sin(rotAngle_rad) + (centroRotatedY_pixels-j) * cos(rotAngle_rad));
		  i_entrada = i_entrada * invScaleX;
		  j_entrada = j_entrada * invScaleY;
		  i_entrada = i_entrada + centroInputX_pixels;
		  j_entrada = centroInputY_pixels - j_entrada;
		  
		  i_entrada = floor(i_entrada);
		  j_entrada = floor(j_entrada);
		  if((i_entrada>=0)&&(i_entrada<sizeInputImage.nPixelsX)&&(j_entrada>=0)&&(j_entrada<sizeInputImage.nPixelsY))
		  {
			value = inputImage->getPixelValue((int)i_entrada, (int)j_entrada, 0);
			rotatedImage->setPixelValue(i, j, 0, value);
		  }
		}
	  }
	}
  }
  else if(interpMethod == BILINEAR)
  {
	for(int i = 0; i < sizeRotatedImage.nPixelsX; i++)
	{
	  for(int j = 0; j < sizeRotatedImage.nPixelsY; j++)
	  {
		rotatedImage->setPixelValue(i, j, 0, 0);
		i_entrada = (i-centroRotatedX_pixels) * cos(rotAngle_rad) + (centroRotatedY_pixels-j) * sin(rotAngle_rad);
		j_entrada = (-(i-centroRotatedX_pixels) * sin(rotAngle_rad) + (centroRotatedY_pixels-j) * cos(rotAngle_rad));
		i_entrada = i_entrada * invScaleX;
		j_entrada = j_entrada * invScaleY;
		i_entrada = i_entrada + centroInputX_pixels;
		j_entrada = centroInputY_pixels - j_entrada;
		// Para la interpolación bilineal, necesito las coordenadas
		i_entrada_ent = floor(i_entrada);
		j_entrada_ent = floor(j_entrada);
		i_entrada_frac = i_entrada - i_entrada_ent;
		j_entrada_frac = j_entrada - j_entrada_ent;
		if((i_entrada>=0)&&(i_entrada<(sizeInputImage.nPixelsX-1))&&(j_entrada>=0)&&(j_entrada<(sizeInputImage.nPixelsY-1)))
		{
		  value = (1-i_entrada_frac)*(1-j_entrada_frac)*inputImage->getPixelValue((int)i_entrada_ent, (int)j_entrada_ent, 0) +
				  (i_entrada_frac)*(1-j_entrada_frac)*inputImage->getPixelValue((int)i_entrada_ent+1, (int)j_entrada_ent, 0) +
				  (1-i_entrada_frac)*(j_entrada_frac)*inputImage->getPixelValue((int)i_entrada_ent, (int)j_entrada_ent+1, 0) +
				  (i_entrada_frac)*(j_entrada_frac)*inputImage->getPixelValue((int)i_entrada_ent+1, (int)j_entrada_ent+1, 0);
		  rotatedImage->setPixelValue(i, j, 0, value);
		}
		else
		{
		  // Dentro de este caso puede estar los casos particulares, de que la interpolación se tenga solo dos píxles
		  // o uno de los 4 necesarios. Como el fov es circular, en realidad los píxeles de la esquina no los necesitaría,
		  // y las filas y columnas de los bordes prácticamente tampoco, solo cerca del centro de la imagen donde el fov
		  // circular llega hasta el extremo de la imagen, pero de todas formas los calculo.
		  value = 0;
		  if(i_entrada==-1)
		  {
			if(j_entrada==-1)
			{
			  value = inputImage->getPixelValue((int)i_entrada_ent+1, (int)j_entrada_ent+1, 0);
			}
			else if(j_entrada==(sizeInputImage.nPixelsY-1))
			{
			  value = inputImage->getPixelValue((int)i_entrada_ent+1, (int)j_entrada_ent, 0);
			}
			else
			{
			  value = (1-j_entrada_frac)*inputImage->getPixelValue((int)i_entrada_ent+1, (int)j_entrada_ent, 0) +
					  (j_entrada_frac)*inputImage->getPixelValue((int)i_entrada_ent+1, (int)j_entrada_ent+1, 0);
			}
		  }
		  else if(i_entrada==(sizeInputImage.nPixelsX-1))
		  {
			if(j_entrada==-1)
			{
			  value = inputImage->getPixelValue((int)i_entrada_ent, (int)j_entrada_ent+1, 0);
			}
			else if(j_entrada==(sizeInputImage.nPixelsY-1))
			{
			  value = inputImage->getPixelValue((int)i_entrada_ent, (int)j_entrada_ent, 0);
			}
			else
			{
			  value = (1-j_entrada_frac)*inputImage->getPixelValue((int)i_entrada_ent, (int)j_entrada_ent, 0) +
					  (j_entrada_frac)*inputImage->getPixelValue((int)i_entrada_ent, (int)j_entrada_ent+1, 0);
			}
		  }
		  else if(j_entrada==-1)
		  {
			value = (1-i_entrada_frac)*inputImage->getPixelValue((int)i_entrada_ent, (int)j_entrada_ent+1, 0) +
					(i_entrada_frac)*inputImage->getPixelValue((int)i_entrada_ent+1, (int)j_entrada_ent+1, 0);
		  }
		  else if(j_entrada==(sizeInputImage.nPixelsX-1))
		  {
			value = (1-i_entrada_frac)*inputImage->getPixelValue((int)i_entrada_ent, (int)j_entrada_ent, 0) +
					(i_entrada_frac)*inputImage->getPixelValue((int)i_entrada_ent+1, (int)j_entrada_ent, 0);
		  }
		  rotatedImage->setPixelValue(i, j, 0, value);
		}
	  }
	}
  }
  else if(interpMethod == BICUBIC)
  {
	for(int i = 0; i < sizeRotatedImage.nPixelsX; i++)
	{
	  for(int j = 0; j < sizeRotatedImage.nPixelsY; j++)
	  {
		rotatedImage->setPixelValue(i, j, 0, 0);
		/*i_entrada = (i-centroX_pixels) * cos(rotAngle_rad) - (centroY_pixels-j) * sin(rotAngle_rad) + centroX_pixels;
		j_entrada = centroY_pixels - ((i-centroX_pixels) * sin(rotAngle_rad) + (centroY_pixels-j) * cos(rotAngle_rad));	
		// Para la interpolación bilineal, necesito las coordenadas
		i_entrada_ent = floor(i_entrada);
		j_entrada_ent = floor(j_entrada);
		i_entrada_frac = i_entrada_ent - i_entrada;
		j_entrada_frac = j_entrada_ent - j_entrada;
		if((i_entrada>=0)&&(i_entrada<sizeImage.nPixelsX)&&(j_entrada>=0)&&(j_entrada<sizeImage.nPixelsY))
		{
		  value = (1-i_entrada_frac)*(1-j_entrada_frac)*inputImage->getPixelValue(i_entrada_ent, j_entrada_ent, 0) +
				  (i_entrada_frac)*(1-j_entrada_frac)*inputImage->getPixelValue(i_entrada_ent+1, j_entrada_ent, 0) +
				  (1-i_entrada_frac)*(j_entrada_frac)*inputImage->getPixelValue(i_entrada_ent, j_entrada_ent+1, 0) +
				  (i_entrada_frac)*(j_entrada_frac)*inputImage->getPixelValue(i_entrada_ent+1, j_entrada_ent+1, 0);
		  (*rotatedImage)->setPixelValue(i, j, 0, value);
		}
		value = inputImage->getPixelValue(round(i_entrada), round(j_entrada), 0);*/
	  }
	}
  }
  return true;
}


bool RotationBasedProjector::Backproject (Sinogram2D* inputProjection, Image* outputImage)
{  
  // Para la backprojection, voy recorriendo cada píxel de la imagen, y lo debo proyectar en la dirección
  // de la lor (o sea según el ángulo de la proyección), y obtener un valor de r, que lo interpolaré entre 
  // dos valores del sinograma.
  Image* rotatedImage;
  float binValue;
  // El tamaño de la iamgen rotada, lo hago igual que numR del sinograma para después directamente
  // sumar las filas:
  SizeImage sizeImage = outputImage->getSize();
  outputImage->fillConstant(0);
  rotatedImage = new Image(sizeImage);
  float centroRotatedX_pixels = ((float)sizeImage.nPixelsX-1) / 2.0f;
  float centroRotatedY_pixels = ((float)sizeImage.nPixelsY-1) / 2.0f;
  // Genero imagen para acotar calculo a fov circular:
  fovImage = new Image(sizeImage);
  float radioFovCuadrado = ((float)sizeImage.nPixelsX*(float)sizeImage.nPixelsX/4.0f);
  for(int i = 0; i < sizeImage.nPixelsX; i++)
  {
	for(int j = 0; j < sizeImage.nPixelsY; j++)
	{
	  if((((float)i)-centroRotatedX_pixels)*((float)i-centroRotatedX_pixels)+(centroRotatedY_pixels-(float)j)*(centroRotatedY_pixels-(float)j) <= radioFovCuadrado)
	  {
		fovImage->setPixelValue(i,j,0, 1);
	  }
	}
  }
  
  // Esto es para un sinograma genérico por lo que no tengo als dimensiones del detector. Entonces
  // los numR de las proyecciones se distribuyen a lo largo de la imagen:
  float r_columna = 0, r1, r2;
  float scaleXtoR = (float)inputProjection->getNumProj() / (float)sizeImage.nPixelsX;
  for(int j = 0; j < sizeImage.nPixelsX; j++)
  {
	for(int k = 0; k < sizeImage.nPixelsY; k++)
	{
	  // Conversión al espacio de proyecciones, depende de numR:
	  r_columna = j * scaleXtoR;
	  r1 = floor(r_columna);
	  r2 = ceil(r_columna);
	  for(int i = 0; i < inputProjection->getNumProj(); i++)
	  {
		outputImage->incrementPixelValue(j,k,0,inputProjection->getAngValue(i));
	  }
	}
  }
  // Voy a rotar la imagen, y luego sumo cada fila para generar el bin de la proyección.
  for(int i = 0; i < inputProjection->getNumProj(); i++)
  {
	// Roto al ángulo de proyección:
	RotateImage(outputImage, rotatedImage, inputProjection->getAngValue(i), this->interpolationMethod);
	// Con la imagen rotada sumo las filas y se las asigno al ángulo i y posR correspondiente a esa fila
	// en el sinograma:
	for(int j = 0; j < sizeImage.nPixelsX; j++)
	{
	  binValue = 0;
	  for(int k = 0; k < sizeImage.nPixelsY; k++)
	  {

	  }
	  // Asigno la suma de esa fila al biin correspondiente:
	  inputProjection->setSinogramBin(i,j,binValue);
	}
  }
  free(rotatedImage);
  return true;
}

/// Sobrecarga que realiza la Backprojection de InputSinogram/EstimatedSinogram
bool RotationBasedProjector::DivideAndBackproject (Sinogram2D* InputSinogram, Sinogram2Dtgs* EstimatedSinogram, Image* outputImage)
{
//  Point2D P1, P2;
//  Line2D LOR;

  return true;
}

bool RotationBasedProjector::Project (Image* inputImage, Sinogram2D* outputProjection)
{
  Image* rotatedImage;
  float binValue;
  // El tamaño de la iamgen rotada, lo hago igual que numR del sinograma para después directamente
  // sumar las filas:
  SizeImage sizeImage;
  sizeImage.nDimensions = 2;
  sizeImage.nPixelsX = outputProjection->getNumR();
  sizeImage.nPixelsY = outputProjection->getNumR();
  sizeImage.nPixelsZ = 1;
  rotatedImage = new Image(sizeImage);
  float centroRotatedX_pixels = ((float)sizeImage.nPixelsX-1) / 2.0f;
  float centroRotatedY_pixels = ((float)sizeImage.nPixelsY-1) / 2.0f;
  // Genero imagen para acotar calculo a fov circular:
  fovImage = new Image(sizeImage);
  float radioFovCuadrado = ((float)sizeImage.nPixelsX*(float)sizeImage.nPixelsX/4.0f);
  for(int i = 0; i < sizeImage.nPixelsX; i++)
  {
	for(int j = 0; j < sizeImage.nPixelsY; j++)
	{
	  if((((float)i)-centroRotatedX_pixels)*((float)i-centroRotatedX_pixels)+(centroRotatedY_pixels-(float)j)*(centroRotatedY_pixels-(float)j) <= radioFovCuadrado)
	  {
		fovImage->setPixelValue(i,j,0, 1);
	  }
	}
  }
  
  // Voy a rotar la imagen, y luego sumo cada fila para generar el bin de la proyección.
  for(int i = 0; i < outputProjection->getNumProj(); i++)
  {
	// Roto al ángulo de proyección:
	RotateImage(inputImage, rotatedImage, outputProjection->getAngValue(i), this->interpolationMethod);
	// Con la imagen rotada sumo las filas y se las asigno al ángulo i y posR correspondiente a esa fila
	// en el sinograma:
	for(int j = 0; j < sizeImage.nPixelsX; j++)
	{
	  binValue = 0;
	  for(int k = 0; k < sizeImage.nPixelsY; k++)
	  {
		if(sqrt((j-centroRotatedX_pixels)*(j-centroRotatedX_pixels)+(centroRotatedY_pixels-k)*(centroRotatedY_pixels-k))< (sizeImage.nPixelsX*sizeImage.nPixelsX/4))
		{
		  binValue += rotatedImage->getPixelValue(j,k,0);
		}
	  }
	  // Asigno la suma de esa fila al biin correspondiente:
	  outputProjection->setSinogramBin(i,j,binValue);
	}
  }
  free(rotatedImage);
  return true;
}

/** Sección para Sinogram3D. */
bool RotationBasedProjector::Project (Image* inputImage, Sinogram3D* outputProjection)
{
  return true;
}

bool RotationBasedProjector::Backproject (Sinogram3D* inputProjection, Image* outputImage)
{
  return true;
}

/// Sobrecarga que realiza la Backprojection del cociente InputSinogram3D/EstimatedSinogram3D
bool RotationBasedProjector::DivideAndBackproject (Sinogram3D* InputSinogram3D, Sinogram3D* EstimatedSinogram3D, Image* outputImage)
{

  return true;
}