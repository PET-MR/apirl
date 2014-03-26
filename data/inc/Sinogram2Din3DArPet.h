/**
	\file Sinogram2Din3DArPet.h
	\brief Archivo que contiene la definición de la clase Sinogram2Din3DArPet.

	Este archivo define la clase Sinogram2Din3DArPet. La misma es una clase derivada
	de Sinogram2D, que extiende la definción de un sinograma 2D dentro
	de un Sinograma 3D del AR-PET. Por esta razón agrega las propiedades de anillo1 y
	anillo 2 que representa el sinograma 2D dentro del michelograma o sinograma 3d. Como cada uno de
	esos sinogramas 2D puede tener polar mashing, o sea que en realidad en un sinograma 2D se juntan
	varias combinaciones de anillos, y se los considera como si fuera uno.
	Tiene en cuenta las cuestions geométricas del aR-PET como por ejemplo, la profundidad del cristal, el 
	tamaño de los cabezales, su resolución, etc.
	\todo 
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2010.10.01
	\version 1.0.0
*/

#ifndef _SINOGRAM2DIN3DARPET_H
#define _SINOGRAM2DIN3DARPET_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <Sinogram2DinCylindrical3Dpet.h>
#include <Utilities.h>

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

/// Clase que define un sinograma de dos dimensiones.
/**
  Clase que define un sinograma de dos dimensiones genérico.
*/
class DLLEXPORT Sinogram2Din3DArPet : public Sinogram2DinCylindrical3Dpet
{
	protected:
	  /** Cantidad de cabezales. */
	  static const int numCabezales = 6;
	  
	  /** Largo del detector en y. */
	  static const float lengthDetectorY_mm = 304.6;
	  
	  /** Largo del detector en x.*/
	  static const float lengthDetectorX_mm = 406.8;
	  
	  /** Espesor del detector, */
	  static const float depthDetector_mm = 25.4;
	  
	  /** Atenuación lineal del detector (en este caso NaI) */
	  static const float linearAttenuationCoef_1_cm = 0.34;
	  
	  /** Distancia del centro del FOV a la superficie de cada detector. */
	  static const float distCenterToDetector_mm = 360;	  
	  
	  /** Largo de zona ciega en el borde de los detectores. */
	  float lengthFromBorderBlindArea_mm;

	  /** Mínima diferencia entre detectores para las coincidencias. Por defecto es 1, o sea que se toman todas las coincidencias. */
	  float minDiffDetectors;
	  //float* getSinogramPtr(){return this->ptrSinogram;};
	  
	  

	public:
	  /// Constructor para cargar los datos de sinogramas a partir del encabezado interfile de un Sinogram2D.
	  /** Constructor para cargar los datos de sinogramas a partir del encabezado interfile de un Sinogram2D como está
		  definido por el stir. Cono en el interfile no está defnido el tamaño del Fov del scanner ni las dimensiones
		  del mismo, se introducen por otro lado dichos parámetros.
		  @param	fileHeaderPath	path completo del archivo de header del sinograma 3d en formato interfile. 
		  @param	rFov_mm radio del FOV en mm.
	  */
	  Sinogram2Din3DArPet(char* fileHeaderPath, float rFov_mm);
	  
	  /** Constructor que recibe como parámetros el tamaño del sinograma
		  y el radio del scanner pensado para scanner cilíndricos. */
	  Sinogram2Din3DArPet(unsigned int nProj, unsigned int nR, float rFov_mm);
	  
	  /** Constructor que realiza la copia de un objeto de esta clase. */
	  Sinogram2Din3DArPet(const Sinogram2Din3DArPet* srcSinogram2Din3DArPet);
	  
	  /// Constructor que realiza la copia de un objeto de la clase Sinogram2DinCylindrical3Dpet.
	  /** Constructor que realiza la copia de un objeto de la clase Sinogram2DinCylindrical3Dpet. 
	   *  Copia las propeidades que tienen en común, como tamaó dle fov y de proyecciones y luego los datos crudos.
	   */
	  Sinogram2Din3DArPet(const Sinogram2DinCylindrical3Dpet* srcSinogram2DinCylindrical3Dpet);
	  
	  /** Constructor que generra un subset de un sinograma. */
	  Sinogram2Din3DArPet(const Sinogram2Din3DArPet* srcSinogram2Din3DArPet, int indexSubset, int numSubsets);
	  
	  /** Destructor. */
	  ~Sinogram2Din3DArPet();
	  
	  /** Método que deveulve una copia del sinograma2d. Se unsa en vez del constructor en las clases derivadas para mantener abstracción.
		@return puntero a un objeto sinograma2d copia del objetco actual.
	  */
	  Sinogram2D* Copy(){ Sinogram2Din3DArPet* sino2dcopy = new Sinogram2Din3DArPet(this); return (Sinogram2D*)sino2dcopy;};
	  
	  /// Función que devuelve las coordenadas de los dos puntos de una LOR.
	  /** Esta función devuelve las coordenadas geomericas globales de los dos puntos que representan una LOR
	   * a partir de las coordenadas del bin del sinograma. Por más que sea un sinograma 2D devuelve un punto 3D
	   * ya que se tiene en cuenta que el sinograma pertence a un conjunto de coordeandas axiales.
	   * \param indexProj índice del ángulo de proyección del sinograma.
	   * \param indexR índice de la distancia r del bin del sinograma deseado.
	   * \param indexRingConfig índice de la combinación de anillos para obtener el ring1 y 2 de la lor. Con
	   * 				este índice se recorre la lista de anillos (si es que hay más de uno).
	   * \param p1 puntero donde devuelve el primer punto 3d que forma la lor.
	   * \param p2 puntero donde se deuelve el segundo punto.
	   * \return devuelve false si no encontró los puntos en los cabezales.
	   */
	  virtual bool getPointsFromLor(int indexProj, int indexR, int indexRingConfig, Point3D* p1, Point3D* p2, float* geomFactor);
	  
	  /// Función que devuelve las coordenadas de los dos puntos de una LOR.
	  /** Esta función devuelve las coordenadas geomericas globales de los dos puntos que representan una LOR
	   * a partir de las coordenadas del bin del sinograma. Por más que sea un sinograma 2D devuelve un punto 3D
	   * ya que se tiene en cuenta que el sinograma pertence a un conjunto de coordeandas axiales.
	   * \param indexProj índice del ángulo de proyección del sinograma.
	   * \param indexR índice de la distancia r del bin del sinograma deseado.
	   * \param p1 puntero donde devuelve el primer punto 2d que forma la lor.
	   * \param p2 puntero donde se deuelve el segundo punto.
	   * \return devuelve false si no encontró los puntos en los cabezales.
	   */
	  bool getPointsFromLor(int indexProj, int indexR, Point2D* p1, Point2D* p2);
	  
	  /** Método que calcula los dos puntos geométricos que forman una Lor.Y adicionalmente devuelve un peso geométrico, según
	    * las características del scanner.
	    @param indexAng índice del ángulo del bin del sinograma a procesar.
	    @param indexR índice de la distancia r del bin del sinograma a procesar.
	    @param p1 puntero a estructura del tipo Point2D donde se guardará el primer punto de la lor (del lado del detector).
	    @param p2 puntero a estructura del tipo Point2D donde se guardará el segundo punto de la lor (del otro lado del detector).
	    @param geomFactor factor geométrico calculado a partir de las coordenadas de la lor.
	    @return devuelve true si encontró los dos puntos sobre el detector, false en caso contrario. 
	  */
	  bool getPointsFromLor (int indexAng, int indexR, Point2D* p1, Point2D* p2, float* geomFactor);
	  
	  /** Método que asigna la mínima diferencia entre detectores para las coincidencias */
	  void setMinDiffDetectors(float minDiff) {minDiffDetectors = minDiff;};
	  
	  /** Método que obtiene la mínima diferencia entre detectores para las coincidencias */
	  float getMinDiffDetectors() {return minDiffDetectors;};
	  
	  /** Método que asigna la mínima diferencia entre detectores para las coincidencias */
	  void setBlindLength(float length_mm) {lengthFromBorderBlindArea_mm = length_mm;};
	  
	  /** Método que obtiene la mínima diferencia entre detectores para las coincidencias */
	  float getBlindLength() {return lengthFromBorderBlindArea_mm;};
};

#endif
