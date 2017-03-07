/**
	\file Segment.h
	\brief Archivo que contiene la definición de la clase abstracta Segment.

	\todo 
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2010.11.11
	\version 1.1.0
*/

#ifndef _SEGMENT_H
#define	_SEGMENT_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <Utilities.h>
#include <Projection.h>
#include <Sinogram2DinCylindrical3Dpet.h>
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

/// Clase que define un segmento dentro de un sinograma 3d.
/**
  Un segmento es un conjunto de sinogramas caracterizados por tener un rango de diferencias de anillos similar.
*/
class Segment
{
  protected:
	/** Número de sinogramas que forman parte de este segmento. */
	int numSinograms;
	/** Máxima diferencia entre anillos para este segmento. */
	int maxRingDiff;	
	/** Mínima diferencia entre anillos para este segmento. */
	int minRingDiff;	
	
	/// Función que inicializa los sinogramas del segmento.
	/** Esta función es abstracta y cada clase derivada debe implementarla según tl tipo
	 * de sinograma 2D que utilice. Debe pedir memoria para el array de sinogramas y luego
	 * para cada uno de ellos.
	 */
	virtual void initSinograms(int nProj, int nR, float rFov_mm, float zFov_mm)=0;
	
	/// Función que inicializa los sinogramas del segmento.
	/** Esta función es abstracta y cada clase derivada debe implementarla según tl tipo
	 * de sinograma 2D que utilice. Debe pedir memoria para el array de sinogramas y luego
	 * para cada uno de ellos.
	 */
	virtual void initSinogramsFromSegment(Segment* srcSegment) =0;
	
  public:
	/// Constructor que inicializa un segmento a partir de todos sus parámetros de dimensiones. 
	/**	Constructor que inicializa un segmento a partir de todos sus parámetros de dimensiones. Ellos
		son las dimensiones del sinograma 2d, los tamaños del fov, la cantidad de sinogramas, y las diferencias
		entre anillos del mismo.
		@param nProj	Número de proyecciones o ángulos de cada sinograma 2d.
		@param nR 	Número de muestras espaciales por proyección del sinograma 2d.
		@param nRings	Número de anillos del scanner.
		@param rFov_mm	Radio del Field of View en mm.
		@param zFov_mm	Largo axial del field of view en mm.
		@param nSinograms	Número de sinogramas2d que contiene este segmento.
		@param nMinRingDiff	Mínima diferencia entre anillos de este segmento.
		@param nMaxRingDiff Máxima diferencia entre anillos de este segmento.
	*/
	Segment(int nProj, int nR, int nRings, float rFov_mm, float zFov_mm, 
	  int nSinograms, int nMinRingDiff, int nMaxRingDiff);
	
	/** Constructor que realiza una copia de un segmento existente.
		@param srcSegment objeto del tipo Segment a partir del cual se creará un nuevo objeto copia de este.
	*/
	Segment(Segment* srcSegmento);
	
	/// Destructor.
	/** Es virtual para que cuando se haga un delete de un objeto de una clase derivada, desde un puntero a esta clase
	 * también se llame al destructor de la clase derivada. */
	virtual ~Segment();
	
	/// Método que devuelve la cantidad de sinogramas de este segmento.
	/**	@return número entero con la cantidad de sinogramas del segmento.
	*/
	int getNumSinograms() {return numSinograms;};
	
	/** Método que deveulve un puntero al Sinograma2D elegido.
		@param indexSegment sinograma que se desea obtener.
		@return puntero a objeto del tipo Sinogram2DinCylindrical3Dpet con el sinograma pedido.
	*/
	virtual Sinogram2DinCylindrical3Dpet* getSinogram2D(int indexSinogram2D) = 0;
	
	/** Método que deveulve un puntero al Sinograma2D elegido.
	    @param sinogram2D sinograma 2D que se desea asignar, debe tener previamente cargados la lista de combinación de anillos que representa dicho sinograma.
	    @param indexInSegment índice dentro del segmento del sinograma.
	    @return true si salió todo bien.
	*/
	virtual bool setSinogram2D(Sinogram2DinCylindrical3Dpet* sinogram2D, int indexInSegment) = 0;
	
	/// Método que devuelve la mínima diferencia de anillos de este segmento.
	/**	@return número entero que representa la mínima diferencia entre anillos.
	*/
	int getMinRingDiff() {return minRingDiff;};
	
	/// Método que devuelve la máxima diferencia de anillos de este segmento.
	/**	@return número entero que representa la máxima diferencia entre anillos.
	*/
	int getMaxRingDiff() {return maxRingDiff;};
	
	
};


#endif