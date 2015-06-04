/**
	\file CuOsemSinogram3d.h
	\brief Archivo que contiene la definición de la clase CuOsemSinogram3d. 
	Clase derivada de OsemSinogram3d, que define el algoritmo Osem para CUDA para sinogramas3D de un cylindrical PET.

	\todo 
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2010.11.11
	\version 1.1.0
*/
#ifndef _CU_OSEMSINOGRAM3D_H_
#define _CU_OSEMSINOGRAM3D_H_

#include <Mlem.h>
#include <CuMlemSinogram3d.h>
#include <Sinogram3D.h>
#include <Logger.h>
#include <omp.h>
#include <cuda.h> // cuda api
#include <cuda_runtime.h>
#include <helper_cuda.h> 	// cuda helper para chequeo de errores
#include <CuSiddonProjector.h>
#include <CuProjector.h>

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
#define DLLEXPORT


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
class DLLEXPORT CuOsemSinogram3d : public CuMlemSinogram3d
{	
  protected:  
    /// Cantidad de subsets que se utilizan para el Osem.
    int numSubsets;
    
    /// Cantidad de proyecciones de cada subset.
    int numProj;
    
    /// Método que calcula la imagen de sensibilidad para uno de los subsets dados.
    /** Método que hace la backprojection de una imagen cosntante para obtener
    la imagen de sensibilidad necesaria para la reconstrucción. */
    bool computeSensitivity(Image*, int indexSubset, TipoProyector tipoProy);
    
    /// Actualiza la imagen en una iteración para un subset.
    /** Actualiza los valores de los píxeles de la imagen reconstruida en una iteración. Es el último paso de la iteración. Depende del subset porque
     * utiliza la sensitivity image. */
    bool updatePixelValue(int subset);
    
    /// Array de imágenes de sensibilidad, una por cada subset.
    /** Tener cuidado con que la memoria que requiere no sea demasiado si los subsets son mucho.
     */
    Image** sensitivityImages;
    
    /// Un umbral de actualización para cada sensitivity image.
    float* updateThresholds;
    
    /// Subsets de la proyección a reconstruir.
    /** Puntero a puntero donde se almacenará un array de sinogramas con los subsets. O sea que d_inputProjectionSubsets[i] es el subset de un sinograma
     * e i va desde 0 a numSubsets-1. */
    float** d_inputProjectionSubsets;
    
    /// Array de punteros para almacenar los ángulos de cada subset..
    /** Array de punteros para almacenar los ángulos de cada subset. Se almacenan en cpu y se copian a gpu antes de cada subiteración. */
    float** h_subsetsAngles;

    /// Puntero a array de Sensitivity Image (una por subset).
    /* Puntero a al dirección de memoria en GPU donde se tendrá la imagen de sensibilidad.*/
    float** d_sensitivityImages;

    /// Inicio la memoria en gpu
    /** Se pide memoria para cada uno de los vectores y se copian los datos de entrada
     */
    bool InitGpuMemory(TipoProyector tipoProy);
    
    /// Inicio las constantes del subset.
    /** Inicia las constantes del subset, que por ahora solo se necesita copiar los ángulos de proyección del subset a la memoria cosntante.
     */
    bool InitSubsetConstants(int indexSubset);
    
    /// Override del likelihood del cumlemsinogramd.
    float getLikelihoodValue(TipoProyector tipoProy);
    
    /// Updates the grid size for processing a subset.
    /** Updates the grid size for processing a subset. The blocksize is fixed, it updates the grid size to process a subset.
     */
    void updateGridSizeForSubsetSinogram();
    
    /// Updates the grid size for processing the whole sinogram.
    /** Updates the grid size for processing the whole sinogram. The blocksize is fixed, it updates the grid size to process a subset.
     */
    void updateGridSizeForWholeSinogram();
    
  public:
    /// Constructores de la clase.
    /* Constructor que carga los parámetros base de una reconstrucción MLEM para Sinogram3D. */
    CuOsemSinogram3d(Sinogram3D* cInputProjection, Image* cInitialEstimate, string cPathSalida, string cOutputPrefix, int cNumIterations, int cSaveIterationInterval, bool cSaveIntermediate, bool cSensitivityImageFromFile, CuProjector* cForwardprojector, CuProjector* cBackprojector, int cNumSubsets);
    
    /// Constructores de la clase a partir de un archivo de configuración.
    /* Constructor que carga los parámetros base de una reconstrucción MLEM
    a partir de un archivo de configuración con cierto formato dado. */
    CuOsemSinogram3d(string configFilename);
    
    
    using Mlem::Reconstruct; // To avoid the warning on possible unintended override.
    /// Método que realiza la reconstrucción de las proyecciones. 
    virtual bool Reconstruct(TipoProyector tipoProy);
    
    /// Método que realiza la reconstrucción y permite al usuario establecer el índice de GPU a utilizar
    virtual bool Reconstruct(TipoProyector tipoProy, int indexGpu);
    
};

#endif
