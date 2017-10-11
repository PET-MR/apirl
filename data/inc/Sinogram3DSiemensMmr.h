/**
	\file Sinogram3DCylindricalPet.h
	\brief Archivo que contiene la definición de la clase Sinogram3DCylindricalPet.

	\todo 
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2010.11.11
	\version 1.1.0
*/

#ifndef _SINOGRAM3DSIEMENSMMR_H
#define	_SINOGRAM3DSIEMENSMMR_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <Utilities.h>
#include <Projection.h>
#include <Interfile.h>
#include <SegmentInCylindrical3Dpet.h>
#include <SegmentInSiemensMmr.h>
#include <Sinogram3D.h>
#include <Sinogram3DCylindricalPet.h>
#include <iostream>
#include <sstream>
#include <fstream>


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

//struct ScannerParameters{
//	/** Radio del Scanner en mm. */
//	const float radioScanner_mm = 656.0f/2.0f;
//	/** Radio del FOV en mm. */
//	const float radioFov_mm = 594.0f/2.0f;
//	/// Size of each pixel element.
//	const float crystalElementSize_mm = 4.0891f;
//	/// Depth or length og each crystal.
//	const float crystalElementLength_mm = 20;
//	/// Mean depth of interaction:
//	const float meanDOI_mm = 9.6f;
//	/// Width of each rings.
//	const float widthRings_mm = 4.0625f; //axialFov_mm / numRings;
//	/** Largo axial del FOV en mm. */
//	const float axialFov_mm = 260.0f; /// widthRings_mm*numRings
//};

/// Clase que define un sinograma de adquisición 3d para el siemens mmr que tiene muchas similtudes que un scanner cilíndrico genérico.
/**Esta clase define un Sinograma 3D practicamente igual al Sinogram3DCylindricalPet, pero reemplazaz las funciones de
 * inicialización de segmentos para usar SegmentInSiemensMmr con Sinogram2DinSiemensMmr.
*/
class DLLEXPORT Sinogram3DSiemensMmr : public Sinogram3DCylindricalPet
{
  private:	
	static const unsigned char sizeBin_bytes = sizeof(float);	// Size in bytes of each element of the Michelogram. float -> 4
  protected:	
	static const struct ScannerParameters scannerParameters;
 	/** Radio del Scanner en mm. */
 	static const float radioScannerMmr_mm;
 	
 	/** Radio del FOV en mm. */
 	static const float radioFovMmr_mm;
 
 	/** Largo axial del FOV en mm. */
 	static const float axialFovMmr_mm;
 	
 	/// Number of cryatl elements including gaps.
 	/** This is amount of crystal elements in the sinogram. The real crystals are 448 gut there are 56 gaps
 	 * counted in the sinograms as crystal elements (1 gap per block). */
 	static const int numCrystalElements = 504;
 	
 	/// Size of each pixel element.
 	static const float crystalElementSize_mm;
 	
 	/** Length of the crystal element. */
 	static const float crystalElementLength_mm;
 	
 	/// Mean depth of interaction:
 	static const float meanDOI_mm;
 	
 	/// Size of num rings.
 	static const int numRings = 64;

	/// Width of each rings.
	static const float widthRingsMmr_mm; //axialFov_mm / numRings;

	/// Función que inicializa los segmentos.
	void inicializarSegmentos();

	/// Setes un sinograma en un segmento.
	void setSinogramInSegment(int indexSegment, int indexSino, Sinogram2DinSiemensMmr* sino2d);
	
  public:	
	/// Constructor para cargar los datos de sinogramas a partir del encabezado interfile de un Sinogram3D.
	/** Constructor para cargar los datos de sinogramas a partir del encabezado interfile de un Sinogram3D como está
		definido por el stir. Cono en el interfile no está defnido el tamaño del Fov del scanner ni las dimensiones
		del mismo, se introducen por otro lado dichos parámetros. El rFov_mm, zFov_mm y rScanner_mm son constantes.
		@param	fileHeaderPath	path completo del archivo de header del sinograma 3d en formato interfile. 
	*/
	Sinogram3DSiemensMmr(char* fileHeaderPath);
	
	/// Constructor que genera un nuevo sinograma3d con un tamaño dado.
	/** Genera un sinograma3d con el tamaño resultante de los aprámetros de entrada. Inicializa los segmentos
	 * y la memoria.
	 * 
	 */
	Sinogram3DSiemensMmr(int numProj, int numR, int numRings, float radioFov_mm, float axialFov_mm, float radioScanner_mm, 
	 int numSegments, int* numSinogramsPerSegment, int* minRingDiffPerSegment, int* maxRingDiffPerSegment);
	
	/// Constructor que genera un nuevo Sinogram3DSiemensMmr que es un subset de un sinograma orginal.
	/** Genera un Sinogram3DSiemensMmr cuyo sinogramas 2d son un subsets de los de un sinograma original.
	 * 
	 */
	Sinogram3DSiemensMmr(Sinogram3DSiemensMmr* srcSinogram3D, int indexSubset, int numSubsets);
	
	/// Constructor para copia desde otro objeto Sinogram3DSiemensMmr.
	/** Constructor que inicializa un nuevo objeto Sinogram3DSiemensMmr a partir de un objeto ya existente. Hace una copia
		de todos los parámetros.
		@param	Sinogram3DSiemensMmr	Sinogram3DSiemensMmr que se copiará en esta nueva instancia. 
	*/
	Sinogram3DSiemensMmr(Sinogram3DSiemensMmr* srcSinogram3D);
	
	///	Destructor.
	~Sinogram3DSiemensMmr();
	
	/** Método que deveulve una copia del sinograma3d. Se unsa en vez del constructor en las clases derivadas para mantener abstracción.
		@return puntero a un objeto sinograma3d copia del objetco actual.
	*/
	Sinogram3D* Copy();

	/// Get subset to a generic sinogram3d.
	/** It cannot be used the generic Sinogram3DCylindricalPet because it allocates the memory
	 * inside the function. */
	Sinogram3D* getSubset(int indexSubset, int numSubsets);
	
	/// Copy all the bins from a source sinograms.
	/** It copies all the bins values from srcSinogram3D into this value.
	 */
	int CopyAllBinsFrom(Sinogram3D* srcSinogram3D);
	
	/** Method that returns the effective radio scanner (takes into account the depth of interction). */
	virtual float getEffectiveRadioScanner_mm(){ return radioScanner_mm + meanDOI_mm;}; //{ return radioScanner_mm + Sinogram3DSiemensMmr::scannerParameters.crystalElementLength_mm/2;};
	  
	/** Gets the crystal size in the transverse direction in mm. */
	virtual float getCrystalElementSize_mm() {return Sinogram3DSiemensMmr::crystalElementSize_mm;}
	/** Gets the length or depth of each crystal in mm. */
	virtual float getCrystalElementLength_mm() {return Sinogram3DSiemensMmr::crystalElementLength_mm;}
	/** Gets the radial bin size in mm. */
	virtual float getRadialBinSize_mm() {return Sinogram3DSiemensMmr::crystalElementSize_mm/2;}	
	/** Gets the mean depth of interactions in mm. */
	virtual float getMeanDOI_mm() {return Sinogram3DSiemensMmr::meanDOI_mm;}
	  
};



#endif