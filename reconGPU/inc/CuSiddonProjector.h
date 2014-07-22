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
#include <Sinogram3DCylindricalPet.h>
#include <CuProjector.h> 
#include <Images.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <vector_types.h>

// DLL export/import declaration: visibility of objects
#ifndef LINK_STATIC
	#ifdef WIN32               // Win32 build
		#ifdef DLL_BUILD    // this applies to DLL building
			#define DLLEXPORT __declspec(dllexport)
		#else                   // this applies to DLL clients/users
			#define DLLEXPORT __declspec(dllimport)
		#endif
		#define DLLLOCAL        // not explicitly export-marked objects are local by default on Win32
	#else
		#ifdef HAVE_GCCVISIBILITYPATCH   // GCC 4.x and patched GCC 3.4 under Linux
			#define DLLEXPORT __attribute__ ((visibility("default")))
			#define DLLLOCAL __attribute__ ((visibility("hidden")))
		#else
			#define DLLEXPORT
			#define DLLLOCAL
		#endif
	#endif
#else                         // static linking
	#define DLLEXPORT
	#define DLLLOCAL
#endif

class DLLEXPORT CuSiddonProjector : virtual CuProjector
{
  private:
	/// Número de lors utilizadas en el siddon por bin del sinograma.
	/** Número de lors utilizadas en el siddon por bin del sinograma. Por default es una lor sola
		que sale del centro del detector. Si se configuraran más son n líneas paralelas equiespaciadas
		sobre el detector por cada bin del sinograma.
		*/
	int numSamplesOnDetector;
	
  public:
	/// Constructor base. 
	/** El cosntructor base setea una lor por bin. */
	CuSiddonProjector();
// 	/** Backprojection con Siddon para Sinogram2D. */
// 	bool Backproject (Sinogram2D* InputSinogram, Image* outputImage);  
// 	/** DivideAndBackprojection con Siddon para Sinogram2D. */
// 	bool DivideAndBackproject (Sinogram2D* InputSinogram, Sinogram2D* EstimatedSinogram, Image* outputImage);
// 	/** Projection con Siddon para Sinogram2D. */
// 	bool Project(Image* image, Sinogram2D* projection);
// 	  
// 	/** Backprojection con Siddon para Sinogram2Dtgs. */
// 	bool Backproject (Sinogram2Dtgs* InputSinogram, Image* outputImage);  
// 	/** DivideAndBackprojection con Siddon para Sinogram2Dtgs. */
// 	bool DivideAndBackproject (Sinogram2Dtgs* InputSinogram, Sinogram2Dtgs* EstimatedSinogram, Image* outputImage);
// 	/** Projection con Siddon para Sinogram2Dtgs. */
// 	bool Project(Image* image, Sinogram2Dtgs* projection);
	
	/** Backprojection con Siddon para Sinogram3D. */
	bool Backproject (float * d_inputSinogram, float* d_outputImage, float *d_ring1, float *d_ring2, Sinogram3DCylindricalPet* inputSinogram, Image* outputImage, bool copyResult); 
	/** DivideAndBackprojection con Siddon para Sinogram3D. */
	bool DivideAndBackproject (float* d_inputSinogram, float* d_estimatedSinogram, float* d_outputImage, float *d_ring1, float *d_ring2, Sinogram3DCylindricalPet* inputSinogram, Image* outputImage, bool copyResult);
	/** Projection con Siddon para Sinogram3D. */
	bool Project (float* d_image, float* d_projection, float *d_ring1, float *d_ring2, Image* inputImage, Sinogram3DCylindricalPet* outputSinogram, bool copyResult);
	
	/// Inicio la memoria en gpu para el proyector
	/** Se pide memoria para cada uno de los vectores y se copian los datos de entrada
	*/
	bool InitGpuMemory(Sinogram3DCylindricalPet* inputSinogram);
};

#endif // PROJECTOR_H