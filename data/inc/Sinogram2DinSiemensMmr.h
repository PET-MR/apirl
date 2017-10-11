/**
	\file Sinogram2DinSiemensMmr.h
	\brief Archivo que contiene la definición de la clase Sinogram2DinCylindrical3Dpet.

	Este archivo define la clase Sinogram2DinSiemensMmr. La misma es una clase derivada
	de Sinogram2DinCylindrical3Dpet, que principalmente sobreescribe la función para obtener las coordenadas de las LORs
	para ajustartse a las características del tomógrafos Siemens Biograph mMr.
	\todo 
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2014.02.10
	\version 1.0.0
*/

#ifndef _SINOGRAM2DINSIEMENSMMR_H
#define _SINOGRAM2DINSIEMENSMMR_H

#include <math.h>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <Sinogram2D.h>
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

/// Clase que define un sinograma para el scanner Siemens Biograph mMr.
/**
  Clase que define un sinograma para el scanner Siemens Biograph mMr.
*/
class DLLEXPORT Sinogram2DinSiemensMmr : public Sinogram2DinCylindrical3Dpet
{
	protected:
	  
	  /// Number of cryatl elements including gaps.
	  /** This is amount of crystal elements in the sinogram. The real crystals are 448 gut there are 56 gaps
	  * counted in the sinograms as crystal elements (1 gap per block). */
	  static const int numCrystalElements = 504;
	  
	  /// Size of each crystal element.
	  static const float crystalElementSize_mm;
	  
	  /** Length of the crystal element. */
	  static const float crystalElementLength_mm;
	public:
	  /// Mean depth of interaction:
	  static const float meanDOI_mm;

	  /// Size of each sinogram's bin.
	  /// It's the size of crystal elemnt divided two (half angles are stored in a different bin). crystalElementSize_mm/2
	  static const float binSize_mm;
	  
	  

	public:
	  /// Constructor para cargar los datos de sinogramas a partir del encabezado interfile de un Sinogram2D.
	  /** Constructor para cargar los datos de sinogramas a partir del encabezado interfile de un Sinogram2D como está
		  definido por el stir. El radio del scanner, del fov y el largo axial del fov se encuentran fijos.
	  */
	  Sinogram2DinSiemensMmr(char* fileHeaderPath);
	  
	  /** Constructor que recibe como parámetros el tamaño del sinograma
		  y el radio del scanner pensado para scanner cilíndricos. */
	  Sinogram2DinSiemensMmr(unsigned int nProj, unsigned int nR);
	  
	  /** Constructor que realiza la copia de un objeto de esta clase. */
	  Sinogram2DinSiemensMmr(const Sinogram2DinSiemensMmr* srcSinogram2DinSiemensMmr);
	  
	  /** Constructor que generra un subset de un sinograma. */
	  Sinogram2DinSiemensMmr(const Sinogram2DinSiemensMmr* srcSinogram2DinSiemensMmr, int indexSubset, int numSubsets);
	  
	  /** Destructor. */
	  ~Sinogram2DinSiemensMmr();
	  
	  /** Método que deveulve una copia del sinograma2d. Se unsa en vez del constructor en las clases derivadas para mantener abstracción.
		@return puntero a un objeto sinograma2d copia del objetco actual.
	  */
	  Sinogram2D* Copy(){ Sinogram2DinSiemensMmr* sino2dcopy = new Sinogram2DinSiemensMmr(this); return (Sinogram2D*)sino2dcopy;};	
	  
	  /** Returns the effective radio scanner, taking into account the depth of interacion.
	   */
	  virtual float getEffectiveRadioScanner_mm(){ return radioScanner_mm + meanDOI_mm;};//{ return (radioScanner_mm + crystalElementLength_mm/2);};
	  /** Método que obtiene los dos puntos límites, de entrada y salida, de una lor que cruza el field of view.
	      El mismo dependerá del tipo de fov del sinograma. Por default es circular, pero
	      puede ser cuadrado o de otra geometría en clases derivadas.
	    @param lor estructura del tipo Line2D que representa una recta que cruza el fov.
	    @param limitPoint1 estructura del tipo Point2D donde se guardarán las coordenadas de un
		    punto donde corta la recta el fov.
	    @param limitPoint2 estructura del tipo Point2D donde se guardarán las coordenadas de un
		    punto donde corta la recta el fov.
	    @return bool devuelve false si la recta no cruza el FoV y true en el caso contrario.
		    Además a través de los parámetros de entrada limitPoint1 y limitPoint2, se devuelven
		    las coordenadas del limite del fov con la recta deseada.
	  */
	  virtual bool getFovLimits(Line2D lor, Point2D* limitPoint1, Point2D* limitPoint2);
	  
	  /** Método que calcula los dos puntos geométricos que forman una Lor. Para el caso de sinograma2d genérico
		  se considera que la lor arranca y termina en el fov, mientras que en un singorama2d de un scanner lo hace
		  sobre los detectores.
		  @param indexAng índice del ángulo del bin del sinograma a procesar.
		  @param indexR índice de la distancia r del bin del sinograma a procesar.
		  @param p1 puntero a estructura del tipo Point2D donde se guardará el primer punto de la lor (del lado del detector).
		  @param p2 puntero a estructura del tipo Point2D donde se guardará el segundo punto de la lor (del otro lado del detector).
		  @return devuelve true si encontró los dos puntos sobre el detector, false en caso contrario. 
	  */
	  bool getPointsFromLor (int indexAng, int indexR, Point2D* p1, Point2D* p2);
	  
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
		//void initParameters(){this->initParameters();}
};

#endif
