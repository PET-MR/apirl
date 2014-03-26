/**
	\file CudaMlem.h
	\brief Archivo que define la clase CudaMlem, derivada de la clase Mlem. 
		   Define una reconstrucción Mlem genérica en cuda agregando ciertos parámetros
		   adicionales necesarios respectos de Mlem. Luego a través de clases derivadas
		   se implementan las reconstrucciones para cada tipo de dato, como por ejemplo
		   en CudaMlemMichelogram.

	\todo 
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2012.03.06
	\version 1.1.0
*/

#ifndef _CUDAMLEM_H_
#define _CUDAMLEM_H_

#include <Images.h>
#include <Mlem.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <iostream>
#include <sstream>
#include <Logger.h>

using namespace::std;

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

/**
    \brief Clase abstracta del método de reconstrucción MLEM.
    Esta clase abstracta define de forma general una reconstrucción del tipo MLEM en cuda. Las clases derivadas
	omplementarán los distintos tipos de reconstrucción, sea 2D, 3D, o con cualquier otra particularidad.Los
	parámetros de reconstrucción son seteados a través de las distintas propiedades de la clase. 
	
    \todo 
*/
class DLLEXPORT CudaMlem : public Mlem
{
  protected:
	dim3 blockSizeProjector;
	dim3 gridSizeProjector;
	dim3 blockSizeBackprojector;
	dim3 gridSizeBackprojector;
	dim3 blockSizeImageUpdate;
	dim3 gridSizeImageUpdate;
  public:
	/// Constructor
	CudaMlem(Image* cInitialEstimate, string cPathSalida, string cOutputPrefix, int cNumIterations, int cSaveIterationInterval, bool cSensitivityImageFromFile, dim3 blockSizeProj, dim3 blockSizeBackproj, dim3 blockSizeIm);
	
	///
	void setExecutionConfigForProjectorKernel(dim3 blockSizeProj);
	void setExecutionConfigForBackrojectorKernel(dim3 blockSizeBackproj);
	void setExecutionConfigForImageKernel(dim3 blockSizeIm);
	
	/// Método que realiza la reconstrucción de las proyecciones. 
	bool Reconstruct();
	
	/// Método que inicializa
	bool initCuda (int, Logger*);
	
};

#endif

