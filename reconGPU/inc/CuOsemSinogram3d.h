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
#include <OsemSinogram3d.h>
#include <Sinogram3D.h>
#include <Logger.h>
#include <omp.h>
#include <cuda.h> // cuda api
#include <helper_cuda.h> 	// cuda helper para chequeo de errores

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
class DLLEXPORT CuOsemSinogram3d : public OsemSinogram3d
{	
  protected:  
    /// Proyección a reconstruir.
    /** Puntero donde se almacenará el sinograma de entrada. */
    float* d_inputProjection;
    
    /// Proyección a reconstruir.
    /** Puntero donde se almacenará el sinograma intermedio de proyección. */
    float* d_estimatedProjection;
    
    /// Imagen a reconstruir en gpu.
    /* Puntero a al dirección de memoria en GPU donde se tendrá la imagen a reconstruir.*/
    float* d_reconstructionImage;
    
    /// Imagen retroproyectada en gpu.
    /* Puntero a al dirección de memoria en GPU donde se tendrá la imagen retroproyectada en cada iteración.*/
    float* d_backprojectedImage;
    
    /// Sensitivity Image.
    /* Puntero a al dirección de memoria en GPU donde se tendrá la imagen de sensibilidad.*/
    float* d_sensitivityImage;
    
    /// Dim3 con configuración de threads per block en cada dimensión para el kernel de proyección.
    dim3 blockSizeProjector;
    
    /// Dim3 con configuración de la grilla en cada dimensión para el kernel de proyección.
    dim3 gridSizeProjector;
    
    /// Dim3 con configuración de threads per block en cada dimensión para el kernel de retroproyección.
    dim3 blockSizeBackprojector;
    
    /// Dim3 con configuración de la grilla en cada dimensión para el kernel de retroproyección.
    dim3 gridSizeBackprojector;
    
    /// im3 con configuración de threads per block en cada dimensión para el kernel de actualización de píxel.
    dim3 blockSizeImageUpdate;
    
    /// Dim3 con configuración de la grilla en cada dimensión para el kernel de píxel.
    dim3 gridSizeImageUpdate;
    
    /// Método que configura los tamaños de ejecución del kernel de proyección.
    void setProjectorKernelConfig(unsigned int numThreadsPerBlockX, unsigned int numThreadsPerBlockY, unsigned int numThreadsPerBlockZ, 
			  unsigned int numBlocksX, unsigned int numBlocksY, unsigned int numBlocksZ);
    
    /// Método que configura los tamaños de ejecución del kernel de retroproyección.
    void setBackprojectorKernelConfig(unsigned int numThreadsPerBlockX, unsigned int numThreadsPerBlockY, unsigned int numThreadsPerBlockZ, 
			  unsigned int numBlocksX, unsigned int numBlocksY, unsigned int numBlocksZ);
    
    /// Método que configura los tamaños de ejecución del kernel de actualización de píxel.
    void setUpdatePixelKernelConfig(unsigned int numThreadsPerBlockX, unsigned int numThreadsPerBlockY, unsigned int numThreadsPerBlockZ, 
			  unsigned int numBlocksX, unsigned int numBlocksY, unsigned int numBlocksZ);
    
    /// Inicio la memoria en gpu
    /** Se pide memoria para cada uno de los vectores y se copian los datos de entrada
     */
    bool InitGpuMemory();
    
    /// Método que copia memoria de cpu en gpu.
    bool CopySinogram3dHostToGpu(float* d_destino, Sinogram3D* h_source);
    
    /// Método que inicializa la gpu.
    bool initCuda (int, Logger*);
    
  public:
    /// Constructores de la clase.
    /* Constructor que carga los parámetros base de una reconstrucción MLEM para Sinogram3D. */
    CuOsemSinogram3d(Sinogram3D* cInputProjection, Image* cInitialEstimate, string cPathSalida, string cOutputPrefix, int cNumIterations, int cSaveIterationInterval, bool cSaveIntermediate, bool cSensitivityImageFromFile, Projector* cForwardprojector, Projector* cBackprojector, int cNumSubsets);
    
    /// Constructores de la clase a partir de un archivo de configuración.
    /* Constructor que carga los parámetros base de una reconstrucción MLEM
    a partir de un archivo de configuración con cierto formato dado. */
    CuOsemSinogram3d(string configFilename);
    
    
    /// Método que realiza la reconstrucción de las proyecciones. 
    bool Reconstruct();
    
    /// Método que realiza la reconstrucción y permite al usuario establecer el índice de GPU a utilizar
    bool Reconstruct(int indexGpu);
    
    
		
};

#endif
