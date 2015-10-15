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

#ifndef _SINOGRAM3DCYLINDRICALPET_H
#define	_SINOGRAM3DCYLINDRICALPET_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <Utilities.h>
#include <Projection.h>
#include <Interfile.h>
#include <SegmentInCylindrical3Dpet.h>
#include <Sinogram3D.h>
#include <iostream>
#include <sstream>
#include <fstream>


// DLL export/import declaration: visibility of objects
#ifndef LINK_STATIC
  #ifdef WIN32               // Win32 build
	#pragma warning( disable: 4251 )
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

/// Clase que define un sinograma de adquisición 3d genérico para un scanner cilíndrico.
/**	Esta clase define un Sinograma 3D en un formato compatible con el stir. Es decir se guardan todos los
	sinogramas 2D en un mismo archivo ordenados según los segmentos. Cada segmento contiene una cantidad N
	de Sinogramas 2D y permite una diferencia mínima y máxima entre anillos. Por esta razón la clase
	tendrá un array de segmentos, y cada uno de ellos contendrá un array de sinogramas.
*/
class DLLEXPORT Sinogram3DCylindricalPet : public Sinogram3D
{
  private:	
	static const unsigned char sizeBin_bytes = sizeof(float);	// Size in bytes of each element of the Michelogram. float -> 4
  protected:
		
	/** Radio del Scanner en mm. */
	float radioScanner_mm;

	/** Array de objetos SegmentInCylindrical3Dpet con los Segmentos que forman el Sinograma 3D. Tiene numSegments elementos. */
	SegmentInCylindrical3Dpet** segments;	
	
	/// Función que inicializa los segmentos.
	void inicializarSegmentos();

	/// Setes un sinograma en un segmento.
	void setSinogramInSegment(int indexSegment, int indexSino, Sinogram2DinCylindrical3Dpet* sino2d);
	
	/** String donde se guardan mensajes del último error ocurrido durante la ejecución. */
	string strError;
  public:	
	/// Constructor para clases derivadas que no pueden utilizar los otros métodos.
	Sinogram3DCylindricalPet(float rFov_mm, float zFov_mm, float rScanner_mm);
	
	/// Constructor para cargar los datos de sinogramas a partir del encabezado interfile de un Sinogram3D.
	/** Constructor para cargar los datos de sinogramas a partir del encabezado interfile de un Sinogram3D como está
		definido por el stir. Cono en el interfile no está defnido el tamaño del Fov del scanner ni las dimensiones
		del mismo, se introducen por otro lado dichos parámetros.
		@param	fileHeaderPath	path completo del archivo de header del sinograma 3d en formato interfile. 
		@param	rFov_mm radio del FOV en mm.
		@param	zFov_mm altura axial del FOV en mm.
		@param	rScanner_mm radio del scanner en mm.
	*/
	Sinogram3DCylindricalPet(char* fileHeaderPath, float rFov_mm, float zFov_mm, float rScanner_mm);
	
	/// Constructor que genera un nuevo sinograma3d que es un subset de un sinograma orginal.
	/** Genera un sinograma3d cuyo sinogramas 2d son un subsets de los de un sinograma original.
	 * 
	 */
	Sinogram3DCylindricalPet(Sinogram3DCylindricalPet* srcSinogram3D, int indexSubset, int numSubsets);
	
	/// Constructor que genera un nuevo sinograma3d con un tamaño dado.
	/** Genera un sinograma3d con el tamaño resultante de los aprámetros de entrada. Inicializa los segmentos
	 * y la memoria.
	 * 
	 */
	Sinogram3DCylindricalPet(int numProj, int numR, int numRings, float radioFov_mm, float axialFov_mm, float radioScanner_mm, 
	 int numSegments, int* numSinogramsPerSegment, int* minRingDiffPerSegment, int* maxRingDiffPerSegment);
	
	/// Constructor para copia desde otro objeto sinograma3d.
	/** Constructor que inicializa un nuevo objeto Sinogram3D a partir de un objeto ya existente. Hace una copia
		de todos los parámetros.
		@param	srcSinogram3D	sinograma3d que se copiará en esta nueva instancia. 
	*/
	Sinogram3DCylindricalPet(Sinogram3DCylindricalPet* srcSinogram3D);
	
	/// Constructor para copia desde otro objeto sinograma3d que solo copia los parámetros.
	/** Constructor que inicializa un nuevo objeto Sinogram3D a partir de un objeto ya existente. Hace una copia
	    de todos los parámetros, pero no incializa los segmentos. El parámetro dummy es solo para diferenciarlo
	    del que si inicializa los segmentos.
	    @param	srcSinogram3D	sinograma3d que se copiará en esta nueva instancia. 
	    @param	dummy	no sirve para nada solo diferencia del otro constructor. 
	*/
	Sinogram3DCylindricalPet(Sinogram3DCylindricalPet* srcSinogram3D, int dummy);
	
	///	Destructor.
	virtual ~Sinogram3DCylindricalPet();
	
	/** Método que deveulve una copia del sinograma3d. Se unsa en vez del constructor en las clases derivadas para mantener abstracción.
		@return puntero a un objeto sinograma3d copia del objetco actual.
	*/
	Sinogram3D* Copy();
	
	/// Otra forma de obtener un subset del sinograma, además de la del constructor específico para ello.
	Sinogram3D* getSubset(int indexSubset, int numSubsets);
	
	/** Método que asigna los tamaños de la geometría con que se adquirió el sinograma3d. Tiene en cuenta
	 *radio y largo del fov, y radio del scanner. 
		@return número de segmentos que contiene el sinograma3d.
	*/
	void setGeometryDim(float rFov_mm, float zFov_mm, float rScanner_mm){ radioFov_mm=rFov_mm; axialFov_mm=zFov_mm; radioScanner_mm=rScanner_mm;};
	
	/** Método que deveulve un pùntero al segmento elegido.
		@param indexSegment segmento que se desea obtener.
		@return puntero a objeto del tipo SegmentInCylindrical3Dpet con el segmento pedido.
	*/
	SegmentInCylindrical3Dpet* getSegment(int indexSegment) {return segments[indexSegment]; };
	
	/// Copy all the bins from a source sinograms.
	/** It copies all the bins values from srcSinogram3D into this value.
	 */
	virtual int CopyAllBinsFrom(Sinogram3D* srcSinogram3D);
	
	/// Copy the multiple ring configuation of each sinogram 2d.
	/** Copy the multiple ring configuation of each sinogram 2d in each segment.
	 */
	int CopyRingConfigForEachSinogram(Sinogram3D* srcSinogram3D);
	
	/** Método que deveulve el radio del scanner cilíndrico. 
		@return radio del cílindro detector del scanner en mm.
	*/
	float getRadioScanner_mm(){ return radioScanner_mm;};
	
	/** Method that returns the effective radio scanner (takes into account the depth of interction). */
	virtual float getEffectiveRadioScanner_mm(){ return radioScanner_mm;};
	
	bool ReadInterfileDiscoverySTE(char* fileDataPath);
	
};



#endif