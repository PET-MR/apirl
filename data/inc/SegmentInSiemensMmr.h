/**
	\file SegmentInCylindrical3Dpet.h
	\brief Archivo que contiene la definición de la clase SegmentInCylindrical3Dpet, que define un segmento dentro de un sinograma 3d.

	\todo Documentar y adaptar nombre de variables. Dejar la estructura similar a la de Sinogram2D.
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2010.11.11
	\version 1.1.0
*/

#ifndef _SEGMENTINSIEMENSMMR_H
#define	_SEGMENTINSIEMENSMMR_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <Utilities.h>
#include <Projection.h>
#include <Sinogram2DinSiemensMmr.h>
#include <SegmentInCylindrical3Dpet.h>
#include <Segment.h>
#include <iostream>

using namespace::std;

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

/// Clase que define un segmento dentro de un sinograma3dSiemensMmr.
/**
  Un segmento es un conjunto de sinogramas caracterizados por tener un rango de diferencias de anillos similar.
*/
class SegmentInSiemensMmr: public SegmentInCylindrical3Dpet
{
  protected:
	/// Función que inicializa los sinogramas del segmento.
	/** Sobreescribe la de SegmentInCylindrical3Dpet para usar Sinogram2DinSiemensMmr.
	 */
	void initSinograms(int nProj, int nR, float rFov_mm, float zFov_mm);
	
	/// Función que inicializa los sinogramas del segmento a partir de los de otro segmento.
	/** 
	 */
	void initSinogramsFromSegment(Segment* srcSegment);
	
	public:
	/// Constructor que inicializa un segmento a partir de todos sus parámetros de dimensiones. 
	/**	Constructor que inicializa un segmento a partir de todos sus parámetros de dimensiones. Ellos
		son las dimensiones del sinograma 2d, los tamaños del fov, la cantidad de sinogramas, y las diferencias
		entre anillos del mismo.
		@param nProj	Número de proyecciones o ángulos de cada sinograma 2d.
		@param nR 	Número de muestras espaciales por proyección del sinograma 2d.
		@param nRings	Número de anillos del scanner.
		@param rFov_mm	Radio del Field of View en mm.
		@param rScanner_mm	Radio del scanner en mm.
		@param nSinograms	Número de sinogramas2d que contiene este segmento.
		@param nMinRingDiff	Mínima diferencia entre anillos de este segmento.
		@param nMaxRingDiff Máxima diferencia entre anillos de este segmento.
	*/
	SegmentInSiemensMmr(int nProj, int nR, int nRings, float rFov_mm, float zFov_mm, float rScanner_mm, 
	  int nSinograms, int nMinRingDiff, int nMaxRingDiff);
	
	/** Constructor que realiza una copia de un segmento existente.
	      @param srcSegment objeto del tipo Segment a partir del cual se creará un nuevo objeto copia de este.
	*/
	SegmentInSiemensMmr(SegmentInSiemensMmr* srcSegmento);
	
	/** Destructor. */
	~SegmentInSiemensMmr();
	
	/// Sets a sinogram2d in the segment.
	/** It cannot use the version of SegmentInCylindrical3Dpet because it has to ask for memory of a Sinogram2DinSiemensMmr.
	 */
	bool setSinogram2D(Sinogram2DinSiemensMmr* sinogram2D, int indexInSegment);
};


#endif