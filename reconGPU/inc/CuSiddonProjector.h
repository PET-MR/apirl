/**
	\file CuSiddonProjector.h
	\brief Archivo que contiene la definición de una clase CuSiddonProjector.

	Esta clase, es una clase derivada de la clase abstracta Projector. Implementa
	la proyección y retroproyección de distintos tipos de datos utilizando como proyector
	Siddon en CUDA.
	
	\todo 
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2010.10.06
	\version 1.1.0
*/

#ifndef _CUSIDDONPROJECTOR_H
#define _CUSIDDONPROJECTOR_H

#include <Sinogram2Dtgs.h>
#include <Sinogram3D.h>
#include <Projector.h> 
#include <Images.h>

class DLLEXPORT CuSiddonProjector : virtual Projector
{
  private:
	/// Número de lors utilizadas en el siddon por bin del sinograma.
	/** Número de lors utilizadas en el siddon por bin del sinograma. Por default es una lor sola
		que sale del centro del detector. Si se configuraran más son n líneas paralelas equiespaciadas
		sobre el detector por cada bin del sinograma.
		*/
	int numSamplesOnDetector;
	
	/// Número de threads per block utilizados en la ejecución.
	unsigned int threads_per_block;
	
	/// Número de threads per block en la dimensión x.
	unsigned int numThreadsPerBlockX;
	/// Número de threads per block en la dimensión y.
	unsigned int numThreadsPerBlockY;
	/// Número de threads per block en la dimensión z.
	unsigned int numThreadsPerBlockZ;
	
	/// Número de bloques en X.
	unsigned int NumberBlocksX;
	/// Número de bloques en Y.
	unsigned int NumberBlocksY;
	/// Número de bloques en Z.
	unsigned int NumberBlocksZ;
	
	
	
  public:
	/// Constructor base. 
	/** El cosntructor base setea una lor por bin. */
	CuSiddonProjector();
	/** Este constructor setea la cantidad de lors por bin que se desea utilizar. */
	CuSiddonProjector(int nSamplesOnDetector);
	/** Backprojection con Siddon para Sinogram2D. */
	bool Backproject (Sinogram2D* InputSinogram, Image* outputImage);  
	/** DivideAndBackprojection con Siddon para Sinogram2D. */
	bool DivideAndBackproject (Sinogram2D* InputSinogram, Sinogram2D* EstimatedSinogram, Image* outputImage);
	/** Projection con Siddon para Sinogram2D. */
	bool Project(Image* image, Sinogram2D* projection);
	  
	/** Backprojection con Siddon para Sinogram2Dtgs. */
	bool Backproject (Sinogram2Dtgs* InputSinogram, Image* outputImage);  
	/** DivideAndBackprojection con Siddon para Sinogram2Dtgs. */
	bool DivideAndBackproject (Sinogram2Dtgs* InputSinogram, Sinogram2Dtgs* EstimatedSinogram, Image* outputImage);
	/** Projection con Siddon para Sinogram2Dtgs. */
	bool Project(Image* image, Sinogram2Dtgs* projection);
	
	/** Backprojection con Siddon para Sinogram3D. */
	bool Backproject (Sinogram3D* InputSinogram, Image* outputImage); 
	/** DivideAndBackprojection con Siddon para Sinogram3D. */
	bool DivideAndBackproject (Sinogram3D* InputSinogram, Sinogram3D* EstimatedSinogram, Image* outputImage);
	/** Projection con Siddon para Sinogram3D. */
	bool Project(Image* image, Sinogram3D* projection);
};

#endif // PROJECTOR_H