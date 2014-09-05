/**
	\file CuProjector.h
	\brief Archivo que contiene la definición de una clase abstracta CuProjector.

	A partir de esta clase abstracta se definen distintos proyectores implementados en cuda. La idea de hacerlo así
	es que CuMlemSinogram3D pueda llamar a través de CuProjector distintos proyectores de la misma manera. Hay un método
	para iniciar la memoria de GPU según el tipo de dato que se use.
	
	\todo 
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2010.10.06
	\version 1.1.0
*/

#include <Projection.h>
#include <Sinogram2Dtgs.h>
#include <Sinogram3D.h>
#include <Images.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <Sinogram3DCylindricalPet.h>

#ifndef _CUPROJECTOR_H
#define _CUPROJECTOR_H

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
/** Clase abstracta CuProjector. */
class DLLEXPORT CuProjector
{
protected:
    /// Número de threads per block utilizados en la ejecución.
    unsigned int threads_per_block;
    
    /// Estructura dim3 con tamaño de grilla.
    dim3 gridSize;
    /// Estructura dim3 con tamaño de bloque.
    dim3 blockSize;
    
    
  public:
    /** Constructor. */
    CuProjector();
    
    /** Constructor que inicializa la configuración de configración del kernel. */
    CuProjector(unsigned int numThreadsPerBlockX, unsigned int numThreadsPerBlockY, unsigned int numThreadsPerBlockZ, 
			  unsigned int numBlocksX, unsigned int numBlocksY, unsigned int numBlocksZ);
// /*	
// 	/** Método abstracto de Project para Sinogram2d. */
//     virtual bool Project(Image*,Sinogram2D*){};
//     /** Método abstracto de Backroject para Sinogram2d. */
//     virtual bool Backproject(Sinogram2D*, Image*){};
//     /** Método abstracto de DivideAndBackproject para Sinogram2d. */
//     virtual bool DivideAndBackproject (Sinogram2D* InputSinogram, Sinogram2D* EstimatedSinogram, Image* outputImage){};
// 	
//     /** Método abstracto de Project para Sinogram2dTgs. */
//     virtual bool Project(Image*,Sinogram2Dtgs*){};
//     /** Método abstracto de Backroject para Sinogram2dTgs. */
//     virtual bool Backproject(Sinogram2Dtgs*, Image*){};
//     /** Método abstracto de DivideAndBackproject para Sinogram2dTgs. */
//     virtual bool DivideAndBackproject (Sinogram2Dtgs* InputSinogram, Sinogram2Dtgs* EstimatedSinogram, Image* outputImage){};
//    */
    /** Backprojection con Siddon para Sinogram3D. */
    virtual bool Backproject (float * d_inputSinogram, float* d_outputImage, int *d_ring1, int *d_ring2, Sinogram3DCylindricalPet* inputSinogram, Image* outputImage, bool copyResult){}; 
    /** DivideAndBackprojection con Siddon para Sinogram3D. */
    virtual bool DivideAndBackproject (float* d_inputSinogram, float* d_estimatedSinogram, float* d_outputImage, int *d_ring1, int *d_ring2, Sinogram3DCylindricalPet* inputSinogram, Image* outputImage, bool copyResult){};
    /** Projection con Siddon para Sinogram3D. */
    virtual bool Project(float* d_image, float* d_projection, int *d_ring1, int *d_ring2, Image* inputImage, Sinogram3DCylindricalPet* outputSinogram, bool copyResult){};
    
    /// Método que incializa parámetros del proyector en la memoria de GPU.
    virtual void initGpuMemory(Sinogram3DCylindricalPet* inputSinogram) {};
    
    /// Método que configura los tamaños de ejecución del kernel.
    void setKernelConfig(unsigned int numThreadsPerBlockX, unsigned int numThreadsPerBlockY, unsigned int numThreadsPerBlockZ, 
			  unsigned int numBlocksX, unsigned int numBlocksY, unsigned int numBlocksZ);
};

#endif // CUPROJECTOR_H
