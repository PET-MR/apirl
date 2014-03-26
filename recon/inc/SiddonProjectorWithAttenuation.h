/**
	\file SiddonProjectorWithAttenuation.h
	\brief Archivo que contiene la definición de una clase SiddonProjectorWithAttenuation.

	Esta clase, es una clase derivada de la clase abstracta Projector. Implementa
	la proyección y retroproyección de distintos tipos de datos utilizando como proyector
	Siddon. Tiene en cuenta la atenuación en el medio.
	
	\todo 
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2010.10.06
	\version 1.1.0
*/

#ifndef _SIDDONPROJECTORWITHATTENUATION_H
#define _SIDDONPROJECTORWITHATTENUATION_H

#include <Sinogram2Dtgs.h>
#include <Sinogram3D.h>
#include <Projector.h> 
#include <Images.h>

class DLLEXPORT SiddonProjectorWithAttenuation : virtual Projector
{
  private:
	/// Número de lors utilizadas en el siddon por bin del sinograma.
	/** Número de lors utilizadas en el siddon por bin del sinograma. Por default es una lor sola
		que sale del centro del detector. Si se configuraran más son n líneas paralelas equiespaciadas
		sobre el detector por cada bin del sinograma.
		*/
	int numSamplesOnDetector;
	
	/// Imagen con mapa de atenuación con coeficientes de atenuación lineal en 1/mm.
	Image* attenuationMap;
	
  public:
	/// Constructor base. 
	/** El cosntructor base setea una lor por bin. */
	SiddonProjectorWithAttenuation(Image* attImage);
	/** Este constructor setea la cantidad de lors por bin que se desea utilizar. */
	SiddonProjectorWithAttenuation(int nSamplesOnDetector, Image* attImage);
	/** Backprojection con Siddon para Sinogram2Dtgs. */
	bool Backproject (Sinogram2Dtgs* InputSinogram, Image* outputImage);  
	/** Projection con Siddon para Sinogram2Dtgs. */
	bool Project(Image* image, Sinogram2Dtgs* projection);
	
	/** Backprojection con Siddon para Sinogram3D. */
	bool Backproject (Sinogram3D* InputSinogram, Image* outputImage); 

	/** Projection con Siddon para Sinogram3D. */
	bool Project(Image* image, Sinogram3D* projection);
	
		
	bool SaveSystemMatrix(Image* inputImage, Sinogram2Dtgs* outputProjection);
};

#endif // PROJECTOR_H