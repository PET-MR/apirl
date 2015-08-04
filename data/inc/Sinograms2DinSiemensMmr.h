/**
	\file Sinograms2DinSiemensMmr.h
	\brief Archivo que contiene la definición de la clase Sinograms2DinSiemensMmr, que define los sinogramas directos 2d 
	para el scanner Siemens mMR.

	\todo 
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2012.11.11
	\version 1.1.0
*/

#ifndef _SINOGRAMS2DINSIEMENSMMR_H
#define	_SINOGRAMS2DINSIEMENSMMR_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <Utilities.h>
#include <Projection.h>
#include <Sinograms2DmultiSlice.h>
#include <Sinogram2DinSiemensMmr.h>
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
    #define DLLLOCAL        // not explicitly export-marked objects are local by default on Win32.
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

/// Clase que define todos los sinogramas directos adquiridos o rebinneados de una adquisición 3d para el scanner SiemensMmr.
/**
  Los sinogramas son todos directos y la cantidad de sinogramas 2d es fija para el sinograma 2d.
*/
class DLLEXPORT Sinograms2DinSiemensMmr : public Sinograms2DmultiSlice
{
  private:
    /// Radio of the mMR.
    static const float radioScanner_mm;
    
    /// Radio of the field of view of the mMR.
    static const float radioFov_mm;
    
    /// Length of the axial field of view of the mMR.
    static const float axialFov_mm;
    
    /// Number of cryatl elements including gaps.
    /** This is amount of crystal elements in the sinogram. The real crystals are 448 gut there are 56 gaps
    * counted in the sinograms as crystal elements (1 gap per block). */
    static const int numCrystalElements = 504;
    
    /// Size of each crystal element.
    static const float crystalElementSize_mm;
    
    /// Size of each sinogram's bin.
    /// It's the size of crystal elemnt divided two (half angles are stored in a different bin). crystalElementSize_mm/2
    static const float binSize_mm;
    
    /// Width of each detector's ring in the mMR.
    static const float widthRings_mm;
    
    /** Sinogramas 2D que forman parte del segmento. 
      Es un vector de punteros a objetos Sinograms2DinSiemensMmr.
    */
    Sinogram2DinSiemensMmr** sinograms2D;	// Array con los sinogramas 2D.
	
  protected:
    /** Método virtual que inicializa los sinogramas 2d. */
    void initSinograms();
  public:
    /** Constructor que crea y levanta los sinogramas 2d para siemens mMR a partir de un archivo interfile. 
      @param fileName nombre del archivo de entrada.
    */
    Sinograms2DinSiemensMmr(string fileName);
    
    /** Constructor que realiza una copia de un set de Sinogram2DinSiemensMmr existente.
      @param srcSegment objeto del tipo Sinograms2DinSiemensMmr a partir del cual se creará un nuevo objeto copia de este.
    */
    Sinograms2DinSiemensMmr(Sinograms2DinSiemensMmr* srcSegmento);
	
    /** Destructor. */
    ~Sinograms2DinSiemensMmr();
    
    
    /** Método que deveulve un puntero al Sinograma2D elegido. Devuelve sinogrm2d para mantener abstracción.
      @param indexSino sinograma que se desea obtener.
      @return puntero a objeto del tipo Sinogram2D con el sinograma pedido.
    */
    Sinogram2D* getSinogram2D(int indexSinogram2D) {return (Sinogram2D*) sinograms2D[indexSinogram2D]; };
    
    /** Obtiene una copia del sinograma actual. */
    Sinograms2DinSiemensMmr* Copy() { Sinograms2DinSiemensMmr* sinoCopia = new Sinograms2DinSiemensMmr(this); return sinoCopia;};
};


#endif
