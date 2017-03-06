#ifndef _SIDDON_H
#define	_SIDDON_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cmath>
#include <Geometry.h>
#include <iostream>
#include <limits>

//using namespace::std;

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

struct SiddonSegment
{
	int IndexX;
	int IndexY;
	int IndexZ;
	float Segment;
};
// This function calculates Siddon Wieghts for a lor. It gets as parameters, the LOR in
// a Line3D object which P0 is the P1 of the LOR, the values of the planes in X, Y, Z, and a pointer
// where all the wieghts will be loaded.
DLLEXPORT float Siddon(Line3D LOR, Image* InputVolume, SiddonSegment** WeightsList, int* LengthList, float factor);

DLLEXPORT float Siddon (Line2D LOR, Image* InputImage, SiddonSegment** WeightsList, int* LengthList, float factor);

DLLEXPORT void Siddon (Point2D point1, Point2D point2, Image* image, SiddonSegment** WeightsList, int* LengthList, float factor);

DLLEXPORT float getRayLengthInFov(Line2D LOR, Image* image);

#endif
