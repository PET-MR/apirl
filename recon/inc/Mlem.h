/**
	\file MLEM.h
	\brief Archivo que contiene la definición de la clase Mlem. 
	Es una clase abstracta que define una clase genérica MLEM, que tiene los elementos
	comunes de este tipo de reconstrucción iterativa: cantidad de iteraciones, nombres
	de salida, proyectores y retroproyectores, etc. Luego las clases derivadas deberán
	considerar las particularidades de algoritmo específico.

	\todo
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2010.11.11
	\version 1.0.0
*/
#pragma once

#include <Images.h>
#include <Geometry.h>
#include <Michelogram.h>
#include <Projection.h>
#include <Projector.h>
#include <Sinogram2D.h>
#include <Sinogram3D.h>
#include <time.h>
#include <string>
#include <iostream>
#include <sstream>

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
#ifdef __cplusplus
//extern "C"
#endif
/*! Enumeración con los tipos de entrada posibles */
typedef enum 
{
	SINOGRAMA2D = 0, SINOGRAMA3D = 1, MICHELOGRAMA = 2, MODO_LISTA = 3
}TiposDatosEntrada;

/**
    \brief Clase abstracta del método de reconstrucción MLEM.
    Esta clase abstracta define de forma general una reconstrucción del tipo MLEM. Las clases derivadas
	omplementarán los distintos tipos de reconstrucción, sea 2D, 3D, o con cualquier otra particularidad.Los
	parámetros de reconstrucción son seteados a través de las distintas propiedades de la clase. 
	
    \todo Por ahora hay un método para cada tipo de dato, lo ideal sería que sea independiente del mismo, 
    que lo único que se requiera sea sea obtener las coordendas de la LOR de un método genérico de las clases
    de proyecciones. Se puede hacer que las clases de tipos de datos, tengan como base una clase del tipo 
    projector, y luego con iterators, y un método que estraiga las coordenadas de una lor abstraernos de la misma.
    También el comando MLEM se lee los parámetros del archivo y luego se los carga en el objeto MLEM, proba
    blemente lo mejor sea que la clase MLEM tenga un método o un constructor que lea directamente los parámetros
    desde el archivo.
	\todo En vez de usar el Logger dentro de esta clase, se puede generar una clase ReconLogger derivada de Logger
	que recibiendo el tamaño de la imagen, el tamaño y tipo de proyección y el tipo de método, automáticamente haga
	el registro de dichos datos.
*/
class DLLEXPORT Mlem
{	
	public:
	  /// Asigna el valor de la propiedad numIterations.
	  /* Asigna el valor a la propiedad numIterations, que indica la cantidad de iteraciones
	  a realizar en la reconstrucción.*/
	  void setNumIterations(int iterations){numIterations = iterations;};
	  
	  /// Asigna el valor de la propiedad outputFilenamePrefix.
	  /* Asigna el valor a la propiedad outputFilenamePrefix, que indica la primera parte (el prefijo)
	  del nombre de los archivos resultantes de la reconstrucción. Tanto archivos de imágenes como de
	  logs.*/
	  void setOutputFilenamePrefix(string prefix){outputFilenamePrefix = prefix;};
	  
	  /// Asigna el valor de la propiedad sensitivityImageFromFile.
	  /* Asigna el valor a la propiedad sensitivityImageFromFile, el cual indica si la sensitivityImage
	  debe ser obtenida desde un archivo o calculada previamente a iniciar la reconstrucción. */
	  void setSensitivityImageFromFile(bool enable){sensitivityImageFromFile = enable;};
	  
	  /// Asigna el valor de la propiedad sensitivityImageFromFile.
	  /* Asigna el valor a la propiedad sensitivityImageFromFile, el cual indica si la sensitivityImage
	  debe ser obtenida desde un archivo o calculada previamente a iniciar la reconstrucción. */
	  void setSensitivityFilename(string filename){sensitivityFilename.assign(filename);};
	  
	  /// Asigna el valor de la propiedad saveIterationInterval.
	  /* Asigna el valor de la propiedad saveIterationInterval para determinar cada cuantas iteraciones
	  se debe guardar la imagen siendo reconstruida. */
	  void setSaveIterationInterval(int numInterval){saveIterationInterval = numInterval;};
	  
	  /// Devuelve la imagen reconstruida
	  /* Devuelve la imagen reconstruida en un objeto del tipo Image. */
	  Image* getReconstructedImage(){return reconstructionImage;};
	  
	  /// Habilita el guardado de la proyección estimada y la imagen retroproyectada dentro de cada iteración.
	  /**  Habilita el guardado de la proyección estimada y la imagen retroproyectada dentro de cada iteración.
		  @param flag flag del tipo bool, que habilita o deshabilita el guardado de los datos intermedios dentro de una iteración.
	  */
	  void enableSaveIntermediates(bool flag){saveIntermediateProjectionAndBackprojectedImage = flag;};
	  
	  /// Devuelve un string con el último mensaje de error.
	  string getLastError();

	protected:	// Protected para que pueda ser utilizadas en clases derivadas como CUDA_MLEM.
	  /// Prefijo del nombre de los archivos de salida.
	  /* Prefijo del nombre de los archivos de salida, básicamente
	     debe identificar la reconstrucción a realizar. A los archivos
	     de salida se los llama con este prefijo, y una identificación
	     de salida para indicar que tipo de reconstrucción es.*/
	  string outputFilenamePrefix;
	  
	  /// Path de salida de los archivos a generar.
	  /* Path de salida de los archivos a generar. */
	  string pathSalida;
	  
	  /// Número de iteraciones.
	  /* Número de iteraciones que se realizarán en la reconstrucción. */
	  int numIterations;
	  
	  /// Imagen inicial.
	  /* En el proceso iterativo se utiliza una imagen inicial, la misma está definida en este
	  objeto Image. Por lo general la misma es una imagen constante, pero puede ser otro tipo
	  de imagen.*/
	  Image* initialEstimate;
	  
	  /// Imagen a reconstruir.
	  /* Esta es la imagen que se utiliza para la reconstrucción, y donde se almacena el resultado
	  final.*/
	  Image* reconstructionImage;
	  
	  /// Sensitivity Image.
	  /* Objeto Image con la imagen de sensibilidad, la misma puede ser calculada o leida
	  de un archivo según la variables generateSensitivityImage. */
	  Image* sensitivityImage;
	  
	  /// Flag que indica si la sensitivity image se lee de un archivo.
	  /* Flag que indica si la sensitivity image se lee de un archivo. */
	  bool sensitivityImageFromFile;
	  
	  /// Nombre del archivo de donde está la sensitivity Image.
	  /* Nombre del archivo de donde está la sensitivity Image, la misma debe estar en formato
	  Interfile, y debe ser del mismo tamaño que la initialEstimate. */
	  string sensitivityFilename;
	  
	  /// Tamaño de la imagen a recnstruir.
	  /* Tamaño de la imagen a reconstruir, si bien el mismo se puede obtener de la reconstructionImage,
	  es cómodo ya tener a mano la estructura con todas las dimensiones. */
	  SizeImage sizeReconImage;
	  
	  /// Intervalo de interaciones en el que su guarda la imagen de reconstrucción.
	  /* Número que indica cada cuantas iteraciones se debe guardar la imagen siendo reconstruida.
	  Por lo tanto, debe ser menor a numIterations. */
	  int saveIterationInterval;
	  
	  /// Flag que habilita el guardado de la proyección estimada y la imagen retroproyectada dentro de cada iteración.
	  /**  Flag que habilita el guardado de la proyección estimada y la imagen retroproyectada dentro de cada iteración.
		  Ambos tipos de datos, los guarda cada saveIterationInterval y en formato interfile.
	  */
	  bool saveIntermediateProjectionAndBackprojectedImage;
	  
	  /// String con el nombre del archivo de log.
	  /* Este string contiene el nombre completo, con path y extensión incluida, del archivo
	  donde se guardará un registro de la reconstrucción. El nombre del mismo será el outputFilenamePrefix
	  seguido de ".log". */
	  string logFileName;
	  
	  /// String donde se guardan los mensajes de error.
	  /* String donde se guarda el último mensaje de error. Se accede desde afuera mediante el
	  método getLastError(). */
	  string strError;
	  
	  /// Array de valores de likelihood.
	  /* Array de valores de likelihood. El array tiene tantas posiciones de memoria
	  como iteraciones. Se le asgina memoria en el proceso de reconstrucción, por lo que
	  para consultarlo se debe haber recontruido una imagen previamente. Se podría hacer
	  en el constructor o en el método setNumIterations, pero tampoco tendría sentido
	  consultarlo antes de una reconstrucción.*/
	  float* likelihoodValues;
	  
	  /// Proyección a reconstruir.
	  /* Objeto del tipo Projection que será la entrada al algoritmo de reconstrucción,
	  puede ser alguno de los distintos tipos de proyección: sinograma 2D, sinograma 3D, etc. */
	  Projection* inputProjection;
	  /// Attenuation Image.
	  /** Objeto Image con la imagen de atenuación con la que se desea realizar la corrección. Puede realizarse con
		 esta imagen o con los factores de corrección directamente.*/
	  Image* attenuationImage;
	  
	  /// Habilitación de corrección por atenuación.
	  /** Flag que habilita la corrección por atenuación, para esto debe estar cargada la attenuationImage
		 y luego calcular la attenuationCorrectionFactorsProjection->
	  */
	  bool enableMultiplicativeTerm;
	  
	  bool enableAdditiveTerm;
	  
	  /// Proyección con factores multiplicativos en el modelo de proyección.
	  /** Objeto del tipo Projection que será la entrada al algoritmo de reconstrucción,
	  puede ser alguno de los distintos tipos de proyección: sinograma 2D, sinograma 3D, etc. 
	  Es el factor multiplicativo en el proyector, o sea debe incluir factores de atenuación y
	  de normalización entre otros. */
	  Projection* multiplicativeProjection;
	  
	  /// Proyección con el término aditivo en la proyección.
	  /** Objeto del tipo Projection que será la entrada al algoritmo de reconstrucción,
	   * puede ser alguno de los distintos tipos de proyección: sinograma 2D, sinograma 3D, etc. 
	   * Este sinograma es un termino aditivo en la proyección por lo que debe incluir corrección por
	   * randoms y scatter. El término aditivo debe estar dividido por el multiplicative factor, 
	   * ya que este se aplica solo en la sensitivity image.
	  */
	  Projection* additiveProjection;
	  
	  /// Umbral de la sensibility image para considerarla en la actualización del píxel.
	  /** En el píxel update, se divide la imagen backprojected por la sensibility image. Esto hace que cuando 
		  los valores de la sensibility image son muy bajos, puede pasar que un píxel empiece a incrementarse
		  cada vez más a medida que pasan las iteraciones, este píxel se va muy fuera de rango y afecta la  
		  visualización de la imagen. Esto se da por lo general en la prfieria. Por esta razón si la sensibility 
		  es muy baja conviente no actualizar el píxel. Vamos a considerar como este umbral el 5% de la escala: 
		  minSens + 0.05(maxSens-minSens)
	  */
	  float updateThreshold;
	  
	  /** Función que actualiza el umbral de actualización de píxel, para esto necesita tener la sensitivity image
		  calculada.
	  */
	  void updateUpdateThreshold();
	  
	protected:
		/// flag que indica si la reconstrucción es 2D, de lo contrario es 3D
		bool recon2D;
		/// Tipo de Dato de Entrada
		TiposDatosEntrada TipoDeDatoEntrada;

		/// Método que calcula el likelihood de las proyecciones. 
		/* Es una función virtual que debe definirse en cada clase derivada específica. */
		float getLikelihoodValue ();
		
		/// Proyector utilizado para hacer la forward projection.
		Projector* forwardprojector;
		
		/// Proyector utilizado para hacer la backprojection.
		Projector* backprojector;

	public:
	  Mlem(){};
	  
	  /// Constructores de la clase base.
	  /* Constructor que carga los parámetros base de una reconstrucción MLEM. */
	  Mlem(Image* cInitialEstimate, string cPathSalida, string cOutputPrefix, int cNumIterations, int cSaveIterationInterval, bool saveIntermediate, bool cSensitivityImageFromFile, Projector* cForwardprojector, Projector* cBackprojector);
	  
	  /// Constructores de la clase a partir de un archivo de configuración.
	  /* Constructor que carga los parámetros base de una reconstrucción MLEM
	  a partir de un archivo de configuración con cierto formato dado. */
	  Mlem(string configFilename);
	  
	  /// Método que carga el mapa de atenuación desde un archivo interfile para aplicar como corrección.
	  /**  Este método habilita la corrección de atenuación y carga la imagen de mapa de atenuación de una imagen interfile.
		  Calcula los attenuation correction factors, que es el logaritmo de la proyección del mapa de atenuación.
		  Se utiliza el forwardproyector como proyector.
	  */
	  virtual bool setAttenuationImage(string attenImageFilename);
	
	  /// Método que carga desde un archivo interfile el factor multiplicativo del modelo de proyección.
	  /**  Este método habilita el factor multiplicativo en el forward model de la proyección.
	  */
	  virtual bool setMultiplicativeProjection(string acfFilename) = 0;
	  
	  /// Método que carga un sinograma desde un archivo interfile con el término aditivo en el modelo de la proyección.
	  /**  Este método habilita el termino aditivo en el forward model del proyector. El término aditivo
	   * debe estar dividido por el multiplicative factor, ya que este se aplica solo en la sensitivity image.
	  */
	  virtual bool setAdditiveProjection(string acfFilename) = 0;
	  
	  /// Método que realiza la reconstrucción de las proyecciones. 
	  virtual bool Reconstruct() {return false;};
		
};
