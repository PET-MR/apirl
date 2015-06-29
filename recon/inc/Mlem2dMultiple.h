/**
	\file Mlem2dMultiple.h
	\brief Archivo que contiene la definición de la clase Mlem2dMultiple. 
	Clase derivada de Mlem2D, que hace la reconstrucción 2d de todos los sinogramas 2d de una adquisición.

	\todo
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2010.11.11
	\version 1.1.0
*/
#ifndef _MLEM2DMULTIPLE_H_
#define _MLEM2DMULTIPLE_H_

#include <Mlem.h>
#include <Mlem2d.h>
#include <Sinograms2DinCylindrical3Dpet.h>
#include <Logger.h>

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



/**
    \brief Clase abstracta del método de reconstrucción MLEM.
    Esta clase abstracta define de forma general una reconstrucción del tipo MLEM. Las clases derivadas
	omplementarán los distintos tipos de reconstrucción, sea 2D, 3D, o con cualquier otra particularidad.Los
	parámetros de reconstrucción son seteados a través de las distintas propiedades de la clase. 
	
    \todo 
*/
class DLLEXPORT Mlem2dMultiple : public Mlem
{	
	  
	  /// Sinogramas a reconstruir.
	  /* Objeto del tipo Projection que será la entrada al algoritmo de reconstrucción,
	  puede ser alguno de los distintos tipos de proyección: sinograma 2D, sinograma 3D, etc. */
	  Sinograms2DmultiSlice* inputProjection;
	  
	  /// Sinograma con factores multiplicativos en el modelo de proyección.
	  /** Objeto del tipo Sinograms2DmultiSlice con el factor multiplicativo en el proyector, o sea debe 
	   * incluir factores de atenuación y de normalización entre otros. 
	   */
	  Sinograms2DmultiSlice* multiplicativeProjection;
	  
	  /// Sinograma con el factor aditivo en el modelo de proyección.
	  /** Objeto del tipo Sinograms2DmultiSlice que será el factor aditivo en la proyección.
	   * Este sinograma es un termino aditivo en la proyección por lo que debe incluir corrección por
	   * randoms y scatter. El término aditivo debe estar dividido por el multiplicative factor, 
	   * ya que este se aplica solo en la sensitivity image.  
	  */
	  Sinograms2DmultiSlice* additiveProjection;
	  
	  /// Método que calcula la imagen de sensibilidad.
	  /* Método que hace la backprojection de una imagen cosntante para obtener
	  la imagen de sensibilidad necesaria para la reconstrucción. */
	  bool computeSensitivity(Image*);
	  
	public:
	  /// Constructores de la clase.
	  /* Constructor que carga los parámetros base de una reconstrucción MLEM 2d para cada sinograma2d adquirido. */
	  Mlem2dMultiple(Sinograms2DmultiSlice* cInputProjection, Image* cInitialEstimate, string cPathSalida, string cOutputPrefix, int cNumIterations, int cSaveIterationInterval, bool cSaveIntermediate, bool cSensitivityImageFromFile, Projector* cForwardprojector, Projector* cBackprojector);
	  
	  /// Constructores de la clase a partir de un archivo de configuración.
	  /* Constructor que carga los parámetros base de una reconstrucción MLEM
	  a partir de un archivo de configuración con cierto formato dado. */
	  Mlem2dMultiple(string configFilename);
	  
	  /// Método que carga desde un archivo interfile el factor multiplicativo del modelo de proyección.
	  /**  Este método habilita el factor multiplicativo en el forward model de la proyección.
	  */
	  bool setMultiplicativeProjection(string acfFilename);
	  
	  /// Método que carga un sinograma desde un archivo interfile con el término aditivo en el modelo de la proyección.
	  /**  Este método habilita el termino aditivo en el forward model del proyector. El término aditivo
	   * debe estar dividido por el multiplicative factor, ya que este se aplica solo en la sensitivity image.
	  */
	  bool setAdditiveProjection(string acfFilename);
	  
	  /// Método que realiza la reconstrucción de las proyecciones. 
	  bool Reconstruct();
	  
	  /// Método que normaliza el volumen reconstruido considerando la sensibilidad de cada slice.
	  bool NormalizeVolume();
		
};

#endif
