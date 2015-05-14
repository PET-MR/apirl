/**
	\file Sinogram3DArPet.h
	\brief Archivo que contiene la definición de la clase Sinogram3DArPet. Es una clase derivada de sinogram3d
		se diferencia que los sinogramas 2d son Sinogram2Din3DArPet.

	\todo 
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2010.11.11
	\version 1.1.0
*/

#ifndef _SINOGRAM3DARPET_H
#define	_SINOGRAM3DARPET_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <Utilities.h>
#include <Projection.h>
#include <Interfile.h>
#include <SegmentIn3DArPet.h>
#include <Sinogram2Din3DArPet.h>
#include <Sinogram3D.h>
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

/// Clase que define un sinograma de adquisición 3d para el ar pet.
/**	Esta clase define un Sinograma 3D en un formato compatible con el stir. Es decir se guardan todos los
	sinogramas 2D en un mismo archivo ordenados según los segmentos. Cada segmento contiene una cantidad N
	de Sinogramas 2D y permite una diferencia mínima y máxima entre anillos. Por esta razón la clase
	tendrá un array de segmentos, y cada uno de ellos contendrá un array de sinogramas.
*/
class DLLEXPORT Sinogram3DArPet : public Sinogram3D 
{
  private:	
	static const unsigned char sizeBin_bytes = sizeof(float);	// Size in bytes of each element of the Michelogram. float -> 4
	
  protected:	  
	/** Array de objetos SegmentIn3DArPet con los Segmentos que forman el Sinograma 3D. Tiene numSegments elementos. */
	SegmentIn3DArPet** segments;	
	
	/// Función que inicializa los segmentos.
	void inicializarSegmentos();
	
	/// Setes un sinograma en un segmento.
	/** Setea un sinograma2d a partir de un Sinogram2DinCylindrical3Dpet. Esto es para poder mantener la abstracción.
	 * 
	 */
	void setSinogramInSegment(int indexSegment, int indexSino, Sinogram2DinCylindrical3Dpet* sino2d);
	
	/** Largo de zona ciega en el borde de los detectores. */
	float lengthFromBorderBlindArea_mm;
	
	/** Mínima diferencia entre detectores para las coincidencias. Por defecto es 1, o sea que se toman todas las coincidencias. */
	float minDiffDetectors;
  public:	
	/// Constructor para cargar los datos de sinogramas a partir del encabezado interfile de un Sinogram3D.
	/** Constructor para cargar los datos de sinogramas a partir del encabezado interfile de un Sinogram3D como está
		definido por el stir. Cono en el interfile no está defnido el tamaño del Fov del scanner ni las dimensiones
		del mismo, se introducen por otro lado dichos parámetros.
		@param	fileHeaderPath	path completo del archivo de header del sinograma 3d en formato interfile. 
		@param	rFov_mm radio del FOV en mm.
		@param	zFov_mm altura axial del FOV en mm.
	*/
	Sinogram3DArPet(char* fileHeaderPath, float rFov_mm, float zFov_mm);
	
	/// Constructor copia.
	/** Constructor copia.
		@param	sino3d	puntero a Sinogram3DArPet del sinograma que se desea copiar. 
	*/
	Sinogram3DArPet(Sinogram3DArPet* sino3d);
	
	/// Constructor que genera un nuevo sinograma3d que es un subset de un sinograma orginal.
	/** Genera un sinograma3d cuyo sinogramas 2d son un subsets de los de un sinograma original.
	 * 
	 */
	Sinogram3DArPet(Sinogram3DArPet* srcSinogram3D, int indexSubset, int numSubsets);
	
	/// Constructor para copia desde otro objeto sinograma3d.
	/** Constructor que inicializa un nuevo objeto Sinogram3D a partir de un objeto ya existente. Hace una copia
		de todos los parámetros.
		@param	srcSinogram3D	sinograma3d que se copiará en esta nueva instancia. 
	*/
	//Sinogram3DArPet(Sinogram3DArPet* srcSinogram3D);
	
	///	Destructor.
	~Sinogram3DArPet();
	
	/** Método que deveulve una copia del sinograma3d. Se unsa en vez del constructor en las clases derivadas para mantener abstracción.
		@return puntero a un objeto sinograma3d copia del objetco actual.
	*/
	Sinogram3D* Copy();
	
	/** Método que asigna los tamaños de la geometría con que se adquirió el sinograma3d. Tiene en cuenta
	 *radio y largo del fov, y radio del scanner. 
		@return número de segmentos que contiene el sinograma3d.
	*/
	void setGeometryDim(float rFov_mm, float zFov_mm){ radioFov_mm=rFov_mm; axialFov_mm=zFov_mm;};
	
	/** Asigna el largo desde el borde de la región de zona ciega en el detector
	 * y se lo asigna a cada uno de los sinogramas 2d que forman parte de este sino3d.
		@return número de segmentos que contiene el sinograma3d.
	*/
	void setLengthOfBlindArea(float length_m);
	
	/** Método que obtiene el largo desde el borde de la zona ciega */
	int getLengthOfBlindArea() {return lengthFromBorderBlindArea_mm;};
	
	/** Método que asigna la mínima diferencia entre detectores para las coincidencias 
	 * y se lo asigna a cada uno de los sinogramas 2d que forman parte de este sino3d.
	 */
	void setMinDiffDetectors(float minDiff);
	
	/** Método que obtiene la mínima diferencia entre detectores para las coincidencias */
	float getMinDiffDetectors() {return minDiffDetectors;};
	
	/** Método que deveulve un pùntero al segmento elegido.
		@param indexSegment segmento que se desea obtener.
		@return puntero a objeto del tipo SegmentInCylindrical3Dpet con el segmento pedido.
	*/
	SegmentIn3DArPet* getSegment(int indexSegment) {return segments[indexSegment]; };
	
	/// Constructor que genera un nuevo sinograma3d que es un subset de un sinograma orginal.
	/** Genera un sinograma3d cuyo sinogramas 2d son un subsets de los de un sinograma original.
	 * 
	 */
	Sinogram3D* getSubset(int indexSubset, int numSubsets);
	
	/// Copy all the bins from a source sinograms.
	/** It copies all the bins values from srcSinogram3D into this value.
	 */
	int CopyAllBinsFrom(Sinogram3D* srcSinogram3D){return 0;};
};



#endif