#ifndef _GEOMETRY_H
#define	_GEOMETRY_H

#include <Images.h>
//#include <Michelogram.h>

#define PI	3.14159265358979323846f; //3.14159265358979323846264338327950288419716939937510;
#define PI_OVER_2	1.57079632679489661923f
#define PI_OVER_4	0.78539816339744830962f
#define DEG_TO_RAD	0.01745329251994329576f
#define RAD_TO_DEG	57.2957795130823208767f

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

struct Point3D
{
	double X;
	double Y;
	double Z;
};

struct Point2D
{
	double X;
	double Y;
};

// Struct that defines a line in a 3D space
// The line parameters are the direction vector (Vx,Vy,Vz)
// and a point that belongs to the line. So the line
// is defined by P = P0 + a*(Vx,Vy,Vz). Also can be written>
// a = (X-X0)/Vx=(Y-Y0)/Vy=(Z-Z0)/Vz
struct Line3D
{
	double Vx;
	double Vy;
	double Vz;
	Point3D P0;
};

// Struct that defines a line in a 2D space
// The line parameters are the direction vector (Vx,Vy)
// and a point that belongs to the line. So the line
// is defined by P = P0 + a*(Vx,Vy). Also can be written>
// a = (X-X0)/Vx=(Y-Y0)/Vy
struct Line2D
{
	double Vx;
	double Vy;
	Point2D P0;
};

// Struct that defines a plane in a 3D space
// The line parameters are the 2 direction vector (Vx,Vy,Vz)
// and (Ux,Uy,Uz) and a point that belongs to the plane. So the plane
// is defined by P = P0 + a*(Vx,Vy,Vz) + b*(Ux,Uy,Uz).
struct PlaneWithVectors
{
	double Ux;
	double Uy;
	double Uz;
	double Vx;
	double Vy;
	double Vz;

	Point3D P0;
};

// Plane defined by the short way: A*x+B*y+C*z+D =0
struct Plane
{
	double A;
	double B;
	double C;
	double D;
};

#ifdef __cplusplus
	extern "C" 
#endif
DLLEXPORT void IntersectionLinePlane(Line3D myLine, Plane myPlane, Point3D* IntersectionPoint);

#ifdef __cplusplus
	extern "C" 
#endif
DLLEXPORT void CalculatePixelsPlanes(Plane* PlanesX, Plane* PlanesY,Plane* PlanesZ, unsigned int SizeX,
					 unsigned int SizeY, unsigned int SizeZ, double RFOV, double ZFOV);
#ifdef __cplusplus
	extern "C" 
#endif
DLLEXPORT void GetPointsFromLOR(double PhiAngle, double r, double Z1, double Z2, double Rscanner,
					  Point3D* P1, Point3D* P2);
DLLEXPORT void GetPointsFromLOR(double PhiAngle, double r, double Rscanner,
				  Point2D* P1, Point2D* P2);
DLLEXPORT void GetPointsFromLOR2 (double PhiAngle, double r, double Z1, double Z2, double Rscanner, Point3D* P1, Point3D* P2);

DLLEXPORT void GetPointsFromTgsLor (float PhiAngle, float r,  float distCentroFrenteCol, float largoCol, Point2D* P1, Point2D* P2);
DLLEXPORT void GetPointsFromTgsLor (float PhiAngle, float r,  float distCentroFrenteCol, float largoCol, float offsetDetector,
						  float offsetCaraColimador, Point2D* P1, Point2D* P2);

#ifdef __cplusplus
	extern "C" 
#endif
DLLEXPORT float distBetweenPoints(Point2D point1, Point2D point2);

// This function generates a 3d projection of a volume, and stores it in
// a Michelogram passed as a parameter. The Michelogram must be filled with
// zeros before calling this function.
/*
#ifdef __cplusplus
	extern "C" 
#endif
DLLEXPORT void Geometric3DProjection (Image* image, Michelogram* MyMichelogram, double Rscanner);
#ifdef __cplusplus
	extern "C" 
#endif
DLLEXPORT void Geometric3DProjectionV2 (Image* image, Michelogram* MyMichelogram, double Rscanner);
*/
#endif
