#pragma once

#include <Images.h>
#include <Geometry.h>
#include <Siddon.h>

#define RSCANNER	41.35f	//radius of PET scanner
#define RADIO_FOV	29.1f
#define AXIAL_FOV 	16.2f
#define CANT_R		192
#define CANT_Z		24
#define DELTA_R		2 * RADIO_FOV/CANT_R	//muestreo espacial (separación en R de cada LOR)
#define DELTA_Z		AXIAL_FOV/CANT_Z
// Pesos para el TriSiddon según la sensibilidad de cada franja dentro de una LOR, ver desarrollo teórico
// de las integrales para más información
#define PESO_CENTRAL	DELTA_R*DELTA_R*2.0f*RSCANNER/2.0f * (1.0f+2.7725f+1.0465f)
#define PESO_LATERAL	DELTA_R*DELTA_R*2.0f*RSCANNER/2.0f * (1.0f+2.1972f)


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

void TriSiddon (float Phi, float R, float Z1, float Z2, SizeImage FOVSize, SiddonSegment** WeightsList, unsigned int* LengthList);
