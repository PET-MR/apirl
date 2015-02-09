/**
	\file OsemSinogram3d.h
	\brief Archivo que contiene la definición de la clase OsemSinogram3d. 
	Clase derivada de MlemSinogram3d, que define el algoritmo Osem para los sinogramas3D de un cylindrical PET.

	\todo 
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2010.11.11
	\version 1.1.0
*/
#ifndef _OSEMSINOGRAM3D_H_
#define _OSEMSINOGRAM3D_H_

#include <Mlem.h>
#include <MlemSinogram3d.h>
#include <Sinogram3D.h>
#include <Logger.h>
#include <omp.h>

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
#ifdef __cplusplus
//extern "C"
#endif
class DLLEXPORT OsemSinogram3d : public MlemSinogram3d
{	
  protected:  
    
    /// Cantidad de subsets que se utilizan para el Osem.
    int numSubsets;
    
    /// Método que calcula la imagen de sensibilidad para uno de los subsets dados.
    /** Método que hace la backprojection de una imagen cosntante para obtener
    la imagen de sensibilidad necesaria para la reconstrucción. */
    bool computeSensitivity(Image*, int indexSubset);
    
    /// Array de imágenes de sensibilidad, una por cada subset.
    /** Tener cuidado con que la memoria que requiere no sea demasiado si los subsets son mucho.
     */
    Image** sensitivityImages;
    
    /// Un umbral de actualización para cada sensitivity image.
    float* updateThresholds;
	  
  public:
    /// Constructores de la clase.
    /** Constructor que carga los parámetros base de una reconstrucción MLEM para Sinogram3D. */
    OsemSinogram3d(Sinogram3D* cInputProjection, Image* cInitialEstimate, string cPathSalida, string cOutputPrefix, int cNumIterations, int cSaveIterationInterval, bool cSaveIntermediate, bool cSensitivityImageFromFile, Projector* cForwardprojector, Projector* cBackprojector, int cNumSubsets);
    
    /// Constructores de la clase a partir de un archivo de configuración.
    /** Constructor que carga los parámetros base de una reconstrucción MLEM
    a partir de un archivo de configuración con cierto formato dado. */
    OsemSinogram3d(string configFilename);
    
    /// Override of the updateUpdateThreshold of the Mlem.
    /** For the osem algorithm, I have one threshold per subset. */
    void updateUpdateThreshold();
    
    /// Método que realiza la reconstrucción de las proyecciones. 
    bool Reconstruct();
		
};

#endif
