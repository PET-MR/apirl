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
	  
	  /// Proyección con factores de corrección por atenuación.
	  /** Objeto del tipo Projection que será la entrada al algoritmo de reconstrucción,
	  puede ser alguno de los distintos tipos de proyección: sinograma 2D, sinograma 3D, etc. */
	  Sinograms2DmultiSlice* attenuationCorrectionFactorsProjection;
	  
	  /// Proyección con la estimación de randoms en cada posición del sinograma.
	  /** Objeto del tipo Projection que será la entrada al algoritmo de reconstrucción,
	   * puede ser alguno de los distintos tipos de proyección: sinograma 2D, sinograma 3D, etc. 
	   * Este sinograma debe restarse al de la adquisición.	  
	  */
	  Sinograms2DmultiSlice* randomsCorrectionProjection;
	  
	  /// Proyección con la estimación del scatter.
	  /** Objeto del tipo Projection que será la entrada al algoritmo de reconstrucción,
	  puede ser alguno de los distintos tipos de proyección: sinograma 2D, sinograma 3D, etc. */
	  Sinograms2DmultiSlice* scatterCorrectionProjection;
	  
	  /// Proyección con factores de corrección para normalización.
	  /** Objeto del tipo Projection que será la entrada al algoritmo de reconstrucción,
	  puede ser alguno de los distintos tipos de proyección: sinograma 2D, sinograma 3D, etc. */
	  Sinograms2DmultiSlice* normalizationCorrectionFactorsProjection;
	  
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
		
		/// Método que carga los coeficientes de corrección de atenuación desde un archivo interfile para aplicar como corrección.
		/**  Este método habilita la corrección de atenuación y carga la imagen de mapa de atenuación de una imagen interfile.
		      
		*/
		bool setAcfProjection(string acfFilename);
		
		/// Método que carga un sinograma desde un archivo interfile con la estimación de scatter para aplicar como corrección.
		/**  Este método habilita la corrección por randoms y carga un sinograma para ello.
		*/
		bool setScatterCorrectionProjection(string acfFilename);
		
		/// Método que carga un sinograma desde un archivo interfile con la estimación de randomc para aplicar como corrección.
		/**  Este método habilita la corrección por randoms y carga un sinograma para ello.
		*/
		bool setRandomCorrectionProjection(string acfFilename);
		
		/// Método que carga un sinograma de normalización desde un archivo interfile.
		/**  Este método habilita la corrección por normalización y carga un sinograma para ello.
		*/
		bool setNormalizationFactorsProjection(string normFilename);
		
		/// Método que aplica las correcciones habilitadas según se hayan cargado los sinogramas de atenuación, randoms y/o scatter.
		bool correctInputSinogram();
		
		/// Método que realiza la reconstrucción de las proyecciones. 
		bool Reconstruct();
		
		/// Método que normaliza el volumen reconstruido considerando la sensibilidad de cada slice.
		bool NormalizeVolume();
		
};

#endif
