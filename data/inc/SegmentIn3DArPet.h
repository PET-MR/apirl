/**
	\file SegmentIn3DArPet.h
	\brief Archivo que contiene la definición de la clase ARPET, que define un segmento dentro de un sinograma 3d del ArPet.

	\todo 
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2010.11.11
	\version 1.1.0
*/

#ifndef _SEGMENTIN3DARPET_H
#define	_SEGMENTIN3DARPET_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <Utilities.h>
#include <Projection.h>
#include <SegmentInCylindrical3Dpet.h>
#include <Sinogram2Din3DArPet.h>
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
class SegmentIn3DArPet : public Segment//, public Sinogram2Din3DArPet
{
  protected:	
    /** Sinogramas 2D que forman parte del segmento. 
	Es un vector de punteros a objetos Sinogram2Din3DArPet.
    */
    Sinogram2Din3DArPet** sinograms2D;	// Array con los sinogramas 2D.
    
    /// Función que inicializa los sinogramas del segmento.
    /** 
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
	    @param nSinograms	Número de sinogramas2d que contiene este segmento.
	    @param nMinRingDiff	Mínima diferencia entre anillos de este segmento.
	    @param nMaxRingDiff Máxima diferencia entre anillos de este segmento.
    */
    SegmentIn3DArPet(int nProj, int nR, int nRings, float rFov_mm, float zFov_mm,  
      int nSinograms, int nMinRingDiff, int nMaxRingDiff);
	
    /** Constructor que realiza una copia de un segmento existente.
	    @param srcSegment objeto del tipo Segment a partir del cual se creará un nuevo objeto copia de este.
    */
    SegmentIn3DArPet(SegmentIn3DArPet* srcSegmento);
    
    /** Constructor que realiza una copia de un segmento existente pero en un segmento SegmentInCylindrical3Dpet, esto me permite mantener compatibilidad.
	    @param srcSegment objeto del tipo Segment a partir del cual se creará un nuevo objeto copia de este.
    */
    SegmentIn3DArPet(SegmentInCylindrical3Dpet* srcSegmento);
    
    /** Destructor. */
    ~SegmentIn3DArPet();
    
    /** Método que deveulve un puntero al Sinograma2D elegido.
	    @param indexSegment sinograma que se desea obtener.
	    @return puntero a objeto del tipo Sinogram2DinCylindrical3Dpet con el sinograma pedido.
    */
    Sinogram2Din3DArPet* getSinogram2D(int indexSinogram2D) {return sinograms2D[indexSinogram2D]; };
    
    /** Método que copia un sinograma2d a uno de los segmentos.
	    @param sinogram2D sinograma 2D que se desea asignar, debe tener previamente cargados la lista de combinación de anillos que representa dicho sinograma.
	    @param indexInSegment índice dentro del segmento del sinograma.
	    @return true si salió todo bien.
    */
    bool setSinogram2D(Sinogram2Din3DArPet* sinogram2D, int indexInSegment);
    
    /** Método que asigna un sinograma2d del tipo cilindrical, es para poder mantener la asbtracción. Lo único que hay que tener en 
     * cuenta es que hay que copiar los datos crudos del sinograma 2d y no copiarlo directamente.
	    @param sinogram2D sinograma 2D que se desea asignar, debe tener previamente cargados la lista de combinación de anillos que representa dicho sinograma.
	    @param indexInSegment índice dentro del segmento del sinograma.
	    @return true si salió todo bien.
    */
    bool setSinogram2D(Sinogram2DinCylindrical3Dpet* sinogram2D, int indexInSegment);
};


#endif