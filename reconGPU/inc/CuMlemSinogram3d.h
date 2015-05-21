/**
	\file CuMlemSinogram3d.h
	\brief Archivo que contiene la definición de la clase CuMlemSinogram3d. 
	Clase derivada de CuMlemSinogram3d, que define el algoritmo Mlem para CUDA para sinogramas3D de un cylindrical PET.

	\todo 
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2010.11.11
	\version 1.1.0
*/
#ifndef _CU_MLEMSINOGRAM3D_H_
#define _CU_MLEMSINOGRAM3D_H_

#include <Mlem.h>
#include <OsemSinogram3d.h>
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



/**
    \brief Clase abstracta del método de reconstrucción MLEM.
    Esta clase abstracta define de forma general una reconstrucción del tipo MLEM. Las clases derivadas
	omplementarán los distintos tipos de reconstrucción, sea 2D, 3D, o con cualquier otra particularidad.Los
	parámetros de reconstrucción son seteados a través de las distintas propiedades de la clase. 
	
    \todo 
*/
    
class DLLEXPORT CuMlemSinogram3d : public MlemSinogram3d
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
    
    /// Puntero a Float en memoria de gpu donde se almacena el valor de likelihhod.
    /** El likelihood estimado es el de d_estimatedProjection en referencia a d_inputProjection.
     */
    float* d_likelihood;
    
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
    
    
    /// Inicio la memoria en gpu
    /** Se pide memoria para cada uno de los vectores y se copian los datos de entrada. Se le indica apra que proyector
     * es porque cambia de proyector en proyector.
     */
    bool InitGpuMemory(TipoProyector tipoProy);
    
    /// Método que copia un sinograma 3d que reside en un objeto del tipo Sinogram3D hacia memoria de gpu.
    int CopySinogram3dHostToGpu(float* d_destino, Sinogram3D* h_source);
    
    /// Método que copia un sinograma 3d en gpu a uno en cpu en un objeto del tipo Sinogram3D.
    int CopySinogram3dGpuToHost(Sinogram3D* h_destino, float* d_source);
    
    /// Método que copia un sinograma 3d que reside en un objeto del tipo Sinogram3D hacia memoria de gpu convirtiéndole en un sinograma sin span.
    /** Con este método se convierte el sinograma3d en memoria de cpu con cierto span, en un sinograma3d sin span en memoria de gpu.
     * Para ello se reparten las cuentas de un sinograma con varias combinaciones axiales en un sinograma por cada combinación de anillo.
     */
    int CopySinogram3dHostToGpuWithoutSpan(float* d_destino, Sinogram3D* h_source);
    
    /// Método que copia un sinograma 3d en gpu a uno en cpu en un objeto del tipo Sinogram3D.
    int CopySinogram3dGpuWithoutSpanToHost(Sinogram3D* h_destino, float* d_source);
    
    /// Método que inicializa la gpu.
    bool initCuda (int, Logger*);
    
    /// Proyector CUDA.
    CuProjector *forwardprojector;
    
    /// Retroproyector CUDA.
    CuProjector *backprojector;
    
    /// Calcula la imagen de sensibilidad.
    /** Depende del tipo de dato utilizado por eso se le debe pasar como parámetro.
     * Utiliza como proyector el backprojector y guarda el resultado en memoria de gpu d_sensitivtyImage y en
     * memoria de cpu sensitivityImage. Además obtiene el umbral de actualización.
     */
    bool computeSensitivity(TipoProyector tipoProy);
    
    /// Realiza la actualización de los valores de píxeles en gpu cada iteracion.
    /** Utiliza la imagen d_backprojectedImage, la d_sensitivityImage y la d_reconstructionImage.
     * El resultado se sobreescribe en d_reconstructionImage.
     */
    bool updatePixelValue();
    
    /// Obtiene el valor de likelihood actual en la reconstrucción.
    /** Estima el likelihood a nivel de gpu entre d_estimatedProjection y d_inputProjection.
     * Guarda el valor en memoria de gpu en d_likelihood y lo devuelve en un float a nivel de cpu.
     */
    float getLikelihoodValue();
    
    /// Copia el resultado actual de la reconstrucciónde GPU a CPU.
    /** La imagen de reconstrucción se actualiza en memoria de gpu en cada iteración en d_reconstructionImage, para poder
     * obtenerla en memoria de cpu se debe llamar a este método que copiará el resultado en reconstructionImage
     */
    void CopyReconstructedImageGpuToHost();
    
    /// Copia reconstructed image en cpu a device.
    /**  Solo debería usarse para iniciar la reconstrucción con alguna iamgen deseada.
     */
    void CopyReconstructedImageHostToGpu();
    
    /// Copia imagen a textura en GPU.
    /** Copia una imagen pasada como parámetro a la única memoria de textura disponible. */
    bool CopyHostImageToTexture(Image* image);
    
    /// Copia desde la memoria de textura en GPU a una imagen en cpu.
    /** Copia una imagen almacenada en la única memoria de textura disponible en GPU 
     *	a una imagen en CPU rexibida como parámetro. */
    bool CopyTextureToHostImage(Image* image);
    
    /// Copia imagen e GPU a textura en GPU.
    /** Copia una imagen en memoria de gpu pasada como parámetro a la única memoria de textura disponible. */
    bool CopyDevImageToTexture(float* d_image, SizeImage imageSize);
    
    /// Copia desde la memoria de textura en GPU a una memoria lineal en gpu.
    /** Copia una imagen almacenada en la única memoria de textura disponible en GPU 
     *	a una imagen en GPU rexibida como parámetro. */
    bool CopyTextureToDevtImage(float* d_image, SizeImage imageSize);
  public:
    /// Constructores de la clase.
    /* Constructor que carga los parámetros base de una reconstrucción MLEM para Sinogram3D. */
    CuMlemSinogram3d(Sinogram3D* cInputProjection, Image* cInitialEstimate, string cPathSalida, string cOutputPrefix, int cNumIterations, int cSaveIterationInterval, bool cSaveIntermediate, bool cSensitivityImageFromFile, CuProjector* cForwardprojector, CuProjector* cBackprojector);
    
    /// Constructores de la clase a partir de un archivo de configuración.
    /* Constructor que carga los parámetros base de una reconstrucción MLEM
    a partir de un archivo de configuración con cierto formato dado. */
    CuMlemSinogram3d(string configFilename);
    
    /// Método que configura los tamaños de ejecución del kernel de proyección.
    /** Recibe como parámetros las dimensiones del bloque, y luego genera el
     * tamaño de grilla según el tamaño de los datos a procesar.
     */
    void setProjectorKernelConfig(unsigned int numThreadsPerBlockX, unsigned int numThreadsPerBlockY, unsigned int numThreadsPerBlockZ);
    
    /// Método que configura los tamaños de ejecución del kernel de retroproyección.
    /** Recibe como parámetros las dimensiones del bloque, y luego genera el
     * tamaño de grilla según el tamaño de los datos a procesar.
     */
    void setBackprojectorKernelConfig(unsigned int numThreadsPerBlockX, unsigned int numThreadsPerBlockY, unsigned int numThreadsPerBlockZ);
    
    /// Método que configura los tamaños de ejecución del kernel de actualización de píxel.
    /** Recibe como parámetros las dimensiones del bloque, y luego genera el
     * tamaño de grilla según el tamaño de los datos a procesar.
     */
    void setUpdatePixelKernelConfig(unsigned int numThreadsPerBlockX, unsigned int numThreadsPerBlockY, unsigned int numThreadsPerBlockZ);
    
    /// Método que configura los tamaños de ejecución del kernel de proyección.
    /** Recibe como parámetro una estructura dim3 con las dimensiones del bloque, y luego genera el
     * tamaño de grilla según el tamaño de los datos a procesar.
     */
    void setProjectorKernelConfig(dim3* blockSize);
    
    /// Método que configura los tamaños de ejecución del kernel de retroproyección.
    /** Recibe como parámetro una estructura dim3 con las dimensiones del bloque, y luego genera el
     * tamaño de grilla según el tamaño de los datos a procesar.
     */
    void setBackprojectorKernelConfig(dim3* blockSize);
    
    /// Método que configura los tamaños de ejecución del kernel de actualización de píxel.
    /** Recibe como parámetro una estructura dim3 con las dimensiones del bloque, y luego genera el
     * tamaño de grilla según el tamaño de los datos a procesar.
     */
    void setUpdatePixelKernelConfig(dim3* blockSize);
    
    using Mlem::Reconstruct; // To avoid the warning on possible unintended override.
    /// Método que realiza la reconstrucción de las proyecciones. 
    virtual bool Reconstruct(TipoProyector tipoProy);
    
    /// Método que realiza la reconstrucción y permite al usuario establecer el índice de GPU a utilizar
    virtual bool Reconstruct(TipoProyector tipoProy, int indexGpu);
};

#endif
