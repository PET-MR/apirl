/**
	\file CuProjectorInterface.h
	\brief Archivo que contiene la definición de una clase CuProjectorInterface.

	Esta clase, es una clase derivada de la clase abstracta Projector. Implementa
	la proyección y retroproyección de distintos tipos de datos utilizando cuda. Este clase sirve como interfaz
	para usar uno de los proyectores CuProjector para hacer proyecciones y retroproyecciones de forma independiente.
	
	\todo 
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2010.10.06
	\version 1.1.0
*/

#ifndef _CUPROJECTORINTERFACE_H
#define _CUPROJECTORINTERFACE_H

#include <Sinogram2Dtgs.h>
#include <Sinogram3D.h>
#include <Projector.h> 
#include <Images.h>
#include <cuda.h> // cuda api
#include <cuda_runtime.h>
#include <helper_cuda.h> 	// cuda helper para chequeo de errores
#include <CuSiddonProjector.h>
#include <CuProjector.h>

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
typedef enum
{
  SIDDON_CYLINDRICAL_SCANNER,
  SIDDON_HEXAGONAL_SCANNER
} TipoProyector;

class DLLEXPORT CuProjectorInterface : virtual Projector
{
  private:
    /// Proyección de salida para proyección o de entrada apra retroproyección.
    /** Puntero donde se almacenará el sinograma de entrada/salida en memoria de gpu. */
    float* d_projection;
    
    /// Imagen de entrada apra proyección o salida para retroproyección.
    /** Puntero a al dirección de memoria en GPU donde se tendrá la imagen de entrada/salida.*/
    float* d_image;
    
    /// CudaArray para manejar la imagen en una textura.
    /** CudaArray para manejar la imagen en una textura 3d. */
    cudaArray *d_imageArray;
    
    /// Array con el índice de ring1 (z1) para cada sinograma 2d del sino3D. En realidad es slice.
    /** El largo del vector es igual a la suma total de sinogramas 2d que tiene el sino3d.Es sólo el índice de anillo!
     * Para obtener la coordenada en el world hay que entrar con este indice al vector de coordenadas de los anillos.
     * El valor es el de slice, por cada anillo físico hay 2*numRings-1 de esa forma se cubren las posiciones intermedias cuando hay span.
     */
    int* d_ring1;
    
    /// Array con el índice de ring1 (z2) para cada sinograma 2d del sino3D.
    /** El largo del vector es igual a la suma total de sinogramas 2d que tiene el sino3d.Es sólo el índice de anillo!
     * Para obtener la coordenada en el world hay que entrar con este indice al vector de coordenadas de los anillos.
     * El valor es el de slice, por cada anillo físico hay 2*numRings-1 de esa forma se cubren las posiciones intermedias cuando hay span.
     */
    int* d_ring2;
    
    /// Array con el índice de ring1 (z1) para cada sinograma 2d del sino3D.
    /** El largo del vector es igual a la suma total de sinogramas 2d que tiene el sino3d.Es sólo el índice de anillo!
     * Para obtener la coordenada en el world hay que entrar con este indice al vector de coordenadas de los anillos.
     */
    float* d_ring1_mm;
    
    /// Array con el índice de ring1 (z2) para cada sinograma 2d del sino3D.
    /** El largo del vector es igual a la suma total de sinogramas 2d que tiene el sino3d.Es sólo el índice de anillo!
     * Para obtener la coordenada en el world hay que entrar con este indice al vector de coordenadas de los anillos.
     */
    float* d_ring2_mm;
    
    /// Dim3 con configuración de threads per block en cada dimensión para el kernel de proyección.
    dim3 blockSizeProjector;
    
    /// Dim3 con configuración de la grilla en cada dimensión para el kernel de proyección.
    dim3 gridSizeProjector;
    
    /// Inicio la memoria en gpu
    /** Se pide memoria para cada uno de los vectores y se copian los datos de entrada. Se le indica apra que proyector
     * es porque cambia de proyector en proyector.
     */
    bool InitGpuMemory(Sinogram3D* sinogram, Image* image, TipoProyector tipoProy);
    
    /// Libera memoria en gpu
    void FreeCudaMemory(void);
    
    /// Método que copia un sinograma 3d que reside en un objeto del tipo Sinogram3D hacia memoria de gpu.
    int CopySinogram3dHostToGpu(float* d_destino, Sinogram3D* h_source);
    
    /// Método que copia un sinograma 3d en gpu a uno en cpu en un objeto del tipo Sinogram3D.
    int CopySinogram3dGpuToHost(Sinogram3D* h_destino, float* d_source);
    
    /// Método que inicializa la gpu.
    bool initCuda (int);
    
    /// Proyector CUDA.
    CuProjector *projector;
    
    /// Type of projector
    TipoProyector typeOfProjector;
    
    /// Gpu Id.
    int gpuId;

  public:
	/// Constructor base. 
	/** El cosntructor base setea una lor por bin. */
	CuProjectorInterface(CuProjector* projector);
	
	/// Método que configura los tamaños de ejecución del kernel de proyección.
	/** Recibe como parámetros las dimensiones del bloque, y luego genera el
	* tamaño de grilla según el tamaño de los datos a procesar.
	*/
	void setProjectorKernelConfig(unsigned int numThreadsPerBlockX, unsigned int numThreadsPerBlockY, unsigned int numThreadsPerBlockZ, Sinogram3D* sinogram);
	
	/// Método que configura los tamaños de ejecución del kernel de proyección.
	/** Recibe como parámetro una estructura dim3 con las dimensiones del bloque, y luego genera el
	* tamaño de grilla según el tamaño de los datos a procesar.
	*/
	void setProjectorKernelConfig(dim3 blockSize, Sinogram3D* sinogram);
	
	/// Método que configura el blocksize para el kernel de proyección.
	/** Método que configura el blocksize para el kernel de proyección. Por defecto en el constructor se utiliza (128,1,1).
	*/
	void setProjectorBlockSizeConfig(dim3 blockSize);
    
	/// Selecciona la Gpu a utilizar.
	bool setGpuId(int id){gpuId = id;};
	
	/** Backprojection con Siddon para Sinogram2D. */
	bool Backproject (Sinogram2D* InputSinogram, Image* outputImage){return false;};  
	/** DivideAndBackprojection con Siddon para Sinogram2D. */
	bool DivideAndBackproject (Sinogram2D* InputSinogram, Sinogram2D* EstimatedSinogram, Image* outputImage){return false;};
	/** Projection con Siddon para Sinogram2D. */
	bool Project(Image* image, Sinogram2D* projection){return false;};
	  
	/** Backprojection con Siddon para Sinogram2Dtgs. */
	bool Backproject (Sinogram2Dtgs* InputSinogram, Image* outputImage){return false;};  
	/** DivideAndBackprojection con Siddon para Sinogram2Dtgs. */
	bool DivideAndBackproject (Sinogram2Dtgs* InputSinogram, Sinogram2Dtgs* EstimatedSinogram, Image* outputImage){return false;};
	/** Projection con Siddon para Sinogram2Dtgs. */
	bool Project(Image* image, Sinogram2Dtgs* projection){return false;};
	
	/** Backprojection con Siddon para Sinogram3D. */
	bool Backproject (Sinogram3D* InputSinogram, Image* outputImage); 
	/** DivideAndBackprojection con Siddon para Sinogram3D. */
	bool DivideAndBackproject (Sinogram3D* InputSinogram, Sinogram3D* EstimatedSinogram, Image* outputImage){};
	/** Projection con Siddon para Sinogram3D. */
	bool Project(Image* image, Sinogram3D* projection);
};

#endif 