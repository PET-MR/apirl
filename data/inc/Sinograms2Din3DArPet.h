/**
	\file Sinograms2Din3DArPet.h
	\brief Archivo que contiene la definición de la clase Sinograms2Din3DArPet, que define los sinogramas directos 2d de un 3d pet.

	\todo Documentar y adaptar nombre de variables. Dejar la estructura similar a la de Sinogram2D.
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2012.11.11
	\version 1.1.0
*/

#ifndef _SINOGRAMS2DIN3DARPET_H
#define	_SINOGRAMS2DIN3DARPET_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <Utilities.h>
#include <Projection.h>
#include <Sinogram2Din3DArPet.h>
#include <Sinograms2DmultiSlice.h>
#include <iostream>
#include <fstream>
#include "../medcon/medcon.h"

using namespace::std;
// DLL export/import declaration: visibility of objects
#ifndef LINK_STATIC
	#ifdef WIN32               // Win32 build
		#ifdef DLL_BUILD    // this applies to DLL building
			#define DLLEXPORT __declspec(dllexport)
		#else                   // this applies to DLL clients/users
			#define DLLEXPORT __declspec(dllimport)
		#endif
		#define DLLLOCAL        // not explicitly export-marked objects are local by default on Win32zz
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

/// Clase que define todos los sinogramas directos adquiridos y luego rebinneados de una adquisición 3d.
/**
  Los sinogramas son todos directos y la cantidad depende del número de anillos o de como se adquiera en cristales continuos.
*/
class Sinograms2Din3DArPet : public Sinograms2DmultiSlice
{
  private:
	Sinogram2Din3DArPet** sinograms2D;	// Array con los sinogramas 2D.
	
	/** Largo de zona ciega en el borde de los detectores. */
	float lengthFromBorderBlindArea_mm;
	
	/** Mínima diferencia entre detectores para las coincidencias. Por defecto es 1, o sea que se toman todas las coincidencias. */
	float minDiffDetectors;
	
  protected:
	void initSinograms();
  public:
	/// Constructor que inicializa los sinogramas 2d con todos sus parámetros de dimensiones. 
	/**	Constructor que inicializa los sinogramas 2d con todos sus parámetros de dimensiones. Ellos
		son las dimensiones del sinograma 2d, los tamaños del fov y la cantidad de sinogramas.
		@param nProj	Número de proyecciones o ángulos de cada sinograma 2d.
		@param nR 	Número de muestras espaciales por proyección del sinograma 2d.
		@param rFov_mm	Radio del Field of View en mm.
		@param zFov_mm	Largo Axial del Field of View en mm. Sirve apra saber la separación entre los nSinogramas
		@param nSinograms	Número de sinogramas2d que contiene este segmento. Puede ser NRings o 2*Nrings+1
	*/
	Sinograms2Din3DArPet(int nProj, int nR, float rFov_mm, float zFov_mm, int nSinograms);
	
	/** Constructor que realiza una copia de un set de sinogramas 2d existente.
		@param srcSegment objeto del tipo Sinograms2DinCylindrical3Dpet a partir del cual se creará un nuevo objeto copia de este.
	*/
	Sinograms2Din3DArPet(Sinograms2Din3DArPet* srcSegmento);
	
	/** Constructor que crea y levanta los sinogramas 2d a partir de un archivo interfile. 
	      @param fileName nombre del archivo de entrada.
	      @param rF_mm radio del fov en mm.
	      @param zF_mm largo axial del fov en mm.
	      @param rSca_mm radio del scanner en mm.
	 */
	Sinograms2Din3DArPet(string fileName, float rF_mm, float zF_mm);
	
	/** Destructor. */
	~Sinograms2Din3DArPet();
	
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
	
	
	/** Método que devuelve un sinograma 2d, devuelve Sinogram2D en vez de Sinograms2Din3DArPet, para mantener abstracción.
	  @param indexSegment sinograma que se desea obtener.
	  @return puntero a objeto del tipo Sinogram2D con el sinograma pedido.
	*/
	Sinogram2Din3DArPet* getSinogram2D(int indexSinogram2D){return sinograms2D[indexSinogram2D]; };
	
	/** Método que devuelve un sinograma 2d, devuelve Sinogram2D en vez de Sinograms2Din3DArPet, para mantener abstracción.
	  @param indexSegment sinograma que se desea obtener.
	  @return puntero a objeto del tipo Sinogram2D con el sinograma pedido.
	*/
	Sinogram2D* getSinogram2DCopy(int indexSinogram2D);
	
	/** Obtiene una copia del sinograma actual. */
	Sinograms2DmultiSlice* Copy() { Sinograms2Din3DArPet* sinoCopia = new Sinograms2Din3DArPet(this); return sinoCopia;};
	
};


#endif
