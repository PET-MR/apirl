#ifndef _UTILITIES_H
#define _UTILITIES_H

#include <limits>

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


struct Event2D
{
	float X1,Y1,X2,Y2;	// Events in a 2D transaxaial plane. 
	// LOR defined by (X1,Y1) and (X2,Y2)
};

struct Event3D
{
	float X1,Y1,Z1,X2,Y2,Z2;	// Events in a 2D transaxaial plane. 
	// LOR defined by (X1,Y1) and (X2,Y2)
};


// Function that searchs the index of the element of an array which
// value is nearer to the parameter Value
DLLEXPORT int SearchBin (float* Array, unsigned int LENGTH, float Value);

DLLEXPORT bool SaveRawFile(void* array, unsigned int BytesElement, unsigned int N, char* path);

#endif
