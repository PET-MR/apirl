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

#define MAX_PHI_VALUES	512	// Máxima cantidad de valores en el angulo theta que puede admitir la implementación.
#define MAX_R_VALUES	512	// Idem para R.
#define MAX_Z_VALUES	92	// Idem para anillos (z)
#define MAX_SPAN	7	// Máximo valor de combinación de anillos por sinograma 2D.auto

// DLL export/import declaration: visibility of objects. Cuda libraries is always static.
/*#ifndef LINK_STATIC
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
#endif*/

#ifdef WIN32               // Win32 build
	#pragma warning( disable: 4251 )
#endif
#define DLLEXPORT

/// Memoria Estática de GPU definido en Otra clase (Debería ser en la de reconstrucción: CuMlem, CuOsem, etc)
// Memoria constante con los valores de los angulos de la proyeccion,
extern  __device__ __constant__ float d_thetaValues_deg[MAX_PHI_VALUES];

// Memoria constante con los valores de la distancia r.
extern __device__ __constant__ float d_RValues_mm[MAX_R_VALUES];

// Memoria constante con los valores de la coordenada axial o z.
extern __device__ __constant__ float d_AxialValues_mm[MAX_Z_VALUES];

// Memoria constante con el radio del scanner (solo para scanner cilíndricos).
extern __device__ __constant__ float d_RadioScanner_mm;

/// Size of each crystal element.
extern __device__ __constant__ float d_crystalElementSize_mm;
/// Size of each sinogram's bin.
extern __device__ __constant__ float d_binSize_mm;
/// Depth or length og each crystal.
extern __device__ __constant__ float d_crystalElementLength_mm;
/// Mean depth of interaction:
extern __device__ __constant__ float d_meanDOI_mm;

// Kernels en la biblioteca.
__global__ void cuSiddonProjection (float* volume, float* michelogram, float *d_ring1, float *d_ring2, int numR, int numProj, int numRings, int numSinos);

__global__ void cuSiddonDivideAndBackproject(float* d_inputSinogram, float* d_estimatedSinogram, float* d_outputImage, 
					     float *d_ring1, float *d_ring2, int numR, int numProj, int numRings, int numSinos);

__global__ void cuSiddonBackprojection(float* d_inputSinogram, float* d_outputImage, 
				       float *d_ring1, float *d_ring2, int numR, int numProj, int numRings, int numSinos);

// Funciones de device en la bilbioteca.
__device__ void CUDA_GetPointsFromLOR (float PhiAngle, float r, float Z1, float Z2, float cudaRscanner, float4* P1, float4* P2);

__device__ void CUDA_GetPointsFromBinsMmr (float PhiAngle, int iR, int numR, float Z1, float Z2, float cudaRscanner, float4* P1, float4* P2);

typedef enum
{
  SIDDON_CYLINDRICAL_SCANNER,
  SIDDON_PROJ_TEXT_CYLINDRICAL_SCANNER,
  SIDDON_BACKPROJ_SURF_CYLINDRICAL_SCANNER,
  SIDDON_HEXAGONAL_SCANNER
} TipoProyector;

// Clase a exportar en la biblioteca.
class DLLEXPORT CuSiddonProjector : virtual CuProjector
{
  private:
	/// Número de lors utilizadas en el siddon por bin del sinograma.
	/** Número de lors utilizadas en el siddon por bin del sinograma. Por default es una lor sola
		que sale del centro del detector. Si se configuraran más son n líneas paralelas equiespaciadas
		sobre el detector por cada bin del sinograma.
		*/
	int numSamplesOnDetector;
	/// Número de lors utilizadas axialmente en por bin del sinograma.
	/** Número de lors utilizadas axialmente en por bin del sinograma. Por default es una lor sola
		que sale del centro del anillo. Si se configuraran más son n líneas paralelas equiespaciadas
		sobre ring correpsondiente a ese sinograma.
		*/
	int numAxialSamplesOnDetector;
  public:
	/// Constructor base. 
	/** El cosntructor base setea una lor por bin. */
	CuSiddonProjector();
	/** Este constructor setea la cantidad de lors por bin que se desea utilizar. */
	CuSiddonProjector(int nSamplesOnDetector);
	/** Este constructor setea la cantidad de lors por bin que se desea utilizar transversalmente y axial. */
	CuSiddonProjector(int nSamplesOnDetector, int nAxialSamplesOnDetector);
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