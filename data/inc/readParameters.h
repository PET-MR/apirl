#ifndef _READPARAMETERS_H
#define	_READPARAMETERS_H

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <ParametersFile.h>
#include <Geometry.h>
#include <string>
#include <math.h>

#define FIXED_KEYS 5

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

/* Encabezados de Funciones relacioandas con la carga de parámetros del Mlem */
DLLEXPORT int getSaveIntermidiateIntervals (string mlemFilename, string cmd, int* saveIterationInterval, bool* saveIntermediateData);
DLLEXPORT int getSensitivityFromFile (string mlemFilename, string cmd, bool* bSensitivityFromFile, string* sensitivityFilename);
DLLEXPORT int getProjectorBackprojectorNames(string mlemFilename, string cmd, string* strForwardprojector, string* strBackprojector);
DLLEXPORT int getSiddonProjectorParameters(string mlemFilename, string cmd, int* numSamples, int* numAxialSamples);
DLLEXPORT int getRotationBasedProjectorParameters(string mlemFilename, string cmd, string *interpMethod);
DLLEXPORT int getCylindricalScannerParameters(string mlemFilename, string cmd, float* radiusFov_mm, float* zFov_mm, float* radiusScanner_mm);
DLLEXPORT int getNumberOfSubsets(string mlemFilename, string cmd, float* numberOfSubsets);
DLLEXPORT int getArPetParameters(string mlemFilename, string cmd, float* radiusFov_mm, float* zFov_mm, float* blindArea_mm, int* minDiffDetectors);
DLLEXPORT int getCorrectionSinogramNames(string mlemFilename, string cmd, string* acfFilename, string* estimatedRandomsFilename, string* estimatedScatterFilename);
DLLEXPORT int getMultiplicativeSinogramName(string mlemFilename, string cmd, string* multiplicativeFilename);
DLLEXPORT int getAdditiveSinogramName(string mlemFilename, string cmd, string* additiveFilename);
#endif
