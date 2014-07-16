#ifndef _CUDA_SIDDON_CUH
#define	_CUDA_SIDDON_CUH

#include <Geometry.h>

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
#ifdef __cplusplus
	extern "C" 
#endif

#define SCANNER_ZFOV	156.96

struct SiddonSegment
{
	unsigned int IndexX;
	unsigned int IndexY;
	unsigned int IndexZ;
	float Segment;
};

typedef enum {
  SENSIBILITY_IMAGE,
  PROJECTION,
  BACKPROJECTION
} SiddonOperation;

#define SENSIBILITY_IMAGE	1
#define PROJECTION		2
#define BACKPROJECTION		3
// This function calculates Siddon Wieghts for a lor. It gets as parameters, the LOR in
// a Line3D object which P0 is the P1 of the LOR, the values of the planes in X, Y, Z, and a pointer
// where all the wieghts will be loaded.
#ifdef __cplusplus
	extern "C" 
#endif
__device__ void CUDA_Siddon (float4* LOR, float4* P0, float* Input, float* Result, int MODE, int indiceMichelogram);
#ifdef __cplusplus
	extern "C" 

