#ifndef _CUDA_GEOMETRY_CUH
#define	_CUDA_GEOMETRY_CUH

#define PI	3.14159265358979323846f; //3.14159265358979323846264338327950288419716939937510;
#define PI_OVER_2	1.57079632679489661923f;
#define PI_OVER_4	0.78539816339744830962f;
#define DEG_TO_RAD	0.01745329251994329576f;
#define RAD_TO_DEG	57.2957795130823208767f;

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

__device__ void CUDA_GetPointsFromLOR (float PhiAngle, float r, float Z1, float Z2, float Rscanner, float4* P1, float4* P2);

#endif
