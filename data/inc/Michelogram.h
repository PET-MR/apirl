#ifndef _MICHELOGRAM_H
#define	_MICHELOGRAM_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <Utilities.h>
#include <cstring>
#include <Projection.h>
#include <Sinogram2DinCylindrical3Dpet.h>
#include <Sinogram3D.h>


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

struct SizeMichelogram
{
	unsigned int NR;
	unsigned int NProj;
	unsigned int NZ;
	float RFOV;			// Radial Dimension of the trnsaxial plane Field of View
	float ZFOV;			// Axial Length of the Field of View
	static const unsigned int MAXANG = 180;
};


class DLLEXPORT Michelogram : virtual Projection
{
	private:
		
		static const unsigned char SIZE_ELEMENT = 4;	// Size in bytes of each element of the Michelogram. float -> 4
	protected:
		
	public:
		Sinogram2DinCylindrical3Dpet** Sinograms2D;	// All Sinograms that take part of the Michelogram
		unsigned char Span;	// Span of the Michelogram
		unsigned char MaxRingDiff;	// Maximum Ring Difference
		float RFOV;			// Radial Dimension of the trnsaxial plane Field of View
		float ZFOV;			// Axial Length of the Field of View
		float rScanner;
		unsigned int NProj;	// Number of Projections (=Number of Angles Values)
		unsigned int NR;	// Number of distance values (spatial sampling)
		unsigned int NZ;	// Number of rings (Axial Sampling)
		unsigned int ZBins;	// Cantidad de bins en el eje Z
		float DistanceBetweenRings;	// Distancia entre anillos.
		float DistanceBetweenBins;	// Distancia entre los bins del eje Z
		//float 
		float* ZValues;		// Vector with the mean value of each ring (Discrete values of the axial axis)
		char Error[200];	// string donde se guarda el �ltimo error ocurrido en alguna operaci�n de esta clase
		
		~Michelogram();

		Michelogram(unsigned int myNProj, unsigned int myNR, unsigned int myNZ, unsigned int mySpan, unsigned int myMaxRingDiff, float myRFOV, float myZFOV);
		Michelogram(SizeMichelogram MySizeMichelogram);
		
		/** Método que calcula el likelihood de esta proyección respecto de una de referencia. */
		float getLikelihoodValue(Projection* referenceProjection){};
		
		bool readFromInterfile(string headerFilename){};
		
		bool Fill(Event3D* Events, unsigned int NEvents);

		bool FillConstant(float Value);

		// Method that reads the Michelogram data from a file. The dimensions of the
		// expected Michelogram are the ones loaded in the constructor of the class
		bool readFromFile(string filePath);

		bool SaveInFile(char* filePath);

		bool ReadDataFromSiemensBiograph16(char* FileName);
		
		bool ReadDataFromSinogram3D(Sinogram3D* MiSino3D);

		float* RawData();
		
		bool FromRawData(float* Raw);
};




struct MichelogramValues
{
	float* PhiValues;
	float* RValues;
	float* ZValues;	
};

void FillValues(MichelogramValues* MyMichelogramValues, SizeMichelogram MySizeMichelogram);

#endif
