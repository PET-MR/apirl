/**
	\file CudaMlemMichelogram.h
	\brief Archivo que contiene la implementación de la clase SiddonProjector.

	\todo 
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2012.03.06
	\version 1.1.0
*/

#ifndef _CUDAMLEMMICHELOGRAM_H_
#define _CUDAMLEMMICHELOGRAM_H_

#include <Images.h>
#include <Geometry.h>
#include <Michelogram.h>
#include <Sinogram3D.h>
#include <CudaMlem.h>
#include <string>
#include <iostream>
#include <sstream>
#include <Logger.h>

using namespace::std;

#define RScanner 886.2/2 //radius of PET scanner
#define NR_MICH	329
#define NPROJ_MICH	280
#define NZ_MICH	24
#define SENSIBILITY_IMAGE	1
#define PROJECTION			2
#define BACKPROJECTION		3
#define MAX_THREADS_PER_BLOCK	512
#define THREADS_PER_BLOCK_SINOGRAM 192	// Threads por bloque para los kernels bin-wise.
//#define DLL_BUILD
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
/*#ifdef __cplusplus
	extern "C" 
#endif*/
// Defino una clase derivada de MLEM, que va a contener m�todos para ejecutar en la CUDA la reconstrucci�n.
class DLLEXPORT CudaMlemMichelogram : public CudaMlem
{
  private:
	/// Proyección a reconstruir.
	/* Objeto del tipo Michelogram que será la entrada al algoritmo de reconstrucción. */
	Michelogram* inputProjection;
	
  public:
	/// Constructor.
	/** Constructor de la clase CudaMlemMichelogram. Recibe los parámetros normales de Mlem, y la configuración de ejecución para cuda. */
	CudaMlemMichelogram(Michelogram* cInputProjection, Image* cInitialEstimate, string cPathSalida, string cOutputPrefix, int cNumIterations, int cSaveIterationInterval, bool cSensitivityImageFromFile, dim3 blockSizeProj, dim3 blockSizeBackproj, dim3 blockSizeIm);

	/// Método que realiza la reconstrucción de las proyecciones. 
	bool Reconstruct();
};
/*
void CUDA_BackProjection (Michelogram* MyMichelogram, Michelogram* ProjectedMichelogram, Volume* BackProjectedVolume, SizeVolume MySizeVolume);


#ifdef __cplusplus
	extern "C" 
#endif
DLLEXPORT void CUDA_SumSystemMatrix(Michelogram* MyMichelogram, SizeVolume MyVolume, float* SumAij);

//#ifdef __cplusplus
//	extern "C" 
//#endif
DLLEXPORT bool CUDA_MLEM(Michelogram* MyMichelogram, Volume* MyVolume, unsigned int N_iterations, char* pathOutput);

//#ifdef __cplusplus
//	extern "C" 
//#endif
DLLEXPORT bool CUDA_MLEM(Michelogram* MyMichelogram, Volume* MyVolume, unsigned int N_iterations, char* pathSensitivityImage, char* pathOutput);

DLLEXPORT bool CUDA_MLEM_Enhanced(Michelogram* MyMichelogram, Volume* MyVolume, unsigned int N_iterations, char* pathOutput);

#ifdef __cplusplus
	extern "C" 
#endif
DLLEXPORT bool CUDA_SaveSiddonPattern(SizeMichelogram MySizeMichelogram, SizeVolume MySizeVolume, char* pathFile);

#ifdef __cplusplus
   extern "C"
#endif
DLLEXPORT bool MyCudaInitialization (void);

#ifdef __cplusplus
   extern "C"
#endif
DLLEXPORT void Init_Geometric_Values (Michelogram* MyMichelogram);

#ifdef __cplusplus
   extern "C"
#endif
DLLEXPORT bool CUDA_SaveSensibilityVolume(SizeMichelogram MySizeMichelogram, SizeVolume MySizeVolume, char* pathFile);
*/

/*#ifdef __cplusplus
   extern "C"
#endif*/
DLLEXPORT bool MyCudaInitialization (int);
DLLEXPORT bool cudaInitialization (int, Logger*);
#endif

