/**
	\file Sinogram2DinCylindrical3Dpet.h
	\brief Archivo que contiene la definición de la clase Sinogram2DinCylindrical3Dpet.

	Este archivo define la clase Sinogram2DinCylindrical3Dpet. La misma es una clase derivada
	de Sinogram2D, que extiende la definción de un sinograma 2D genérico a un sinograma 2D dentro
	de un Sinograma 3D de un scanner cilíndrico. Por esta razón agrega las propiedades de anillo1 y
	anillo 2 que representa el sinograma 2D dentro del michelograma o sinograma 3d. Como cada uno de
	esos sinogramas 2D puede tener polar mashing, o sea que en realidad en un sinograma 2D se juntan
	varias combinaciones de anillos, y se los considera como si fuera uno.
	\todo 
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2010.10.01
	\version 1.0.0
*/

#ifndef _SINOGRAM2DINCYLINDRICAL3DPET_H
#define _SINOGRAM2DINCYLINDRICAL3DPET_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <Sinogram2D.h>
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
class DLLEXPORT Sinogram2DinCylindrical3Dpet : public Sinogram2D
{
	protected:
	  /** Radio del scanner. */
	  float radioScanner_mm;	  
	  
	  /** Números de anillos que representa este sinograma 2D. Tiene utilidad para cuando al
		  sinograma 2d se le quiere dar un contexto en el espacio tridimensional. */
	  int ring1, ring2;	
	  
	  /** Cantidad de LORs representadas por cada bin del sinograma. Este parámetro tiene
		  sentido cuando es un sinograma 2d dentro de uno 3d, y hay polar mashing. */
	  unsigned int numZ;
	  /** Vectores con los anillos 1 y 2 para cada lor representada para cualquier bin de este
		  sinograma. Tiene tantos elementos como numZ. */
	  int* ptrListRing1, *ptrListRing2;	
	  
	  /** Vectores con las posiciones axiales en mmm de los anillos 1 y 2 para cada lor representada para cualquier bin de este
		  sinograma. Tiene tantos elementos como numZ. */
	  float* ptrListZ1_mm, *ptrListZ2_mm;
	  //float* getSinogramPtr(){return this->ptrSinogram;};
	  

	public:
	  Sinogram2DinCylindrical3Dpet(){};
	  /// Constructor para cargar los datos de sinogramas a partir del encabezado interfile de un Sinogram2D.
	  /** Constructor para cargar los datos de sinogramas a partir del encabezado interfile de un Sinogram2D como está
		  definido por el stir. Cono en el interfile no está defnido el tamaño del Fov del scanner ni las dimensiones
		  del mismo, se introducen por otro lado dichos parámetros.
		  @param	fileHeaderPath	path completo del archivo de header del sinograma 3d en formato interfile. 
		  @param	rFov_mm radio del FOV en mm.
		  @param	rScanner_mm radio del FOV en mm.
	  */
	  Sinogram2DinCylindrical3Dpet(char* fileHeaderPath, float rFov_mm, float rScanner_mm);
	  
	  /** Constructor que recibe como parámetros el tamaño del sinograma
		  y el radio del scanner pensado para scanner cilíndricos. */
	  Sinogram2DinCylindrical3Dpet(unsigned int nProj, unsigned int nR, float rFov_mm, float rScanner_mm);
	  
	  /** Constructor que realiza la copia de un objeto de esta clase. */
	  Sinogram2DinCylindrical3Dpet(const Sinogram2DinCylindrical3Dpet* srcSinogram2DinCylindrical3Dpet);
	  
	  /** Constructor que generra un subset de un sinograma. */
	  Sinogram2DinCylindrical3Dpet(const Sinogram2DinCylindrical3Dpet* srcSinogram2DinCylindrical3Dpet, int indexSubset, int numSubsets);
	  
	  /** Destructor. */
	  virtual ~Sinogram2DinCylindrical3Dpet();
	  
	  /** Método que deveulve una copia del sinograma2d. Se unsa en vez del constructor en las clases derivadas para mantener abstracción.
		@return puntero a un objeto sinograma2d copia del objetco actual.
	  */
	  virtual Sinogram2D* Copy(){ Sinogram2DinCylindrical3Dpet* sino2dcopy = new Sinogram2DinCylindrical3Dpet(this); return (Sinogram2D*)sino2dcopy;};	
	  
	  /** Método que devuelve cantidad de combinación de anillos que representa este sinograma.
		  Solo se tiene en cuenta cuando es parte de un sinograma 3d y hay polar mashing.
	  */
	  unsigned int getNumZ(){ return numZ;};
	  
	  /** Método que devuelve el valor del anillo 1 correspondiente a este sinograma. */
	  int getRing1(){ return ring1;};
	  
	  /** Método que devuelve el valor del anillo 2 correspondiente a este sinograma. */
	  int getRing2(){ return ring2;};
	  
	  /** Método que setea el valor del anillo 1 correspondiente a este sinograma. */
	  void setRing1(int r1){ ring1 = r1;};
	  
	  /** Método que setea el valor del anillo 2 correspondiente a este sinograma. */
	  void setRing2(int r2){ ring2 = r2;};
	  
	  /** Método que setea el valor del radio del scanner en mm a este sinograma. */
	  void setRadioScanner(float rScanner_mm){ radioScanner_mm = rScanner_mm;};
	  
	  /** Método que devuelve el valor del radio del scanner en mm. */
	  float getRadioScanner_mm(){ return radioScanner_mm;};
	  
	  /** Método que setea una configuración de combinación de ángulos polares para este sinograma 2D.
		  O sea, define cuantas combinaciones de anillos representa cada bin del sinograma, y la lista de combinación
		  de ellos.
		  @param nZ número de combinación de anillos o angulos posibles a definir.
		  @param listRing1 lista de valores de anillos para el anillo 1, debe tener nZ elementos.
		  @param listRing2 lista de valores de anillos para el anillo 2, debe tener nZ elementos.
		  @param listZ1 lista de valores axiales en mm de anillos para el anillo 1, debe tener nZ elementos.
		  @param listZ2 lista de valores axiales en mm de anillos para el anillo 2, debe tener nZ elementos.
	  */
	  void setMultipleRingConfig(int nZ, int* listRing1, int* listRing2, float* listZ1, float* listZ2);
	  
	  /** Método que copia la configuración para múltiples anillos de otro sinograma 2d.
		@param srcSinogram sinograma de donde se sacará la configuración de anillos.
	  */
	  void copyMultipleRingConfig(Sinogram2DinCylindrical3Dpet* srcSinogram);
	  
	  /** Método que devuelve el valor index para el anillo 1 de las listas de combinaciones de anillos
	    para este sinograma. Esto es válido cuando está dentro de un sinograma 3d, y hay polar mashing. 
	    @param index índice de las distintas combinaciones de anillos o valores de z que representa cad abin del sinograma.
	    @return valor del anillo 1 para el valor index de las distintas combinaciones polares del sinograma.	
	  */
	  int getRing1FromList(int index){ return ptrListRing1[index];};
	  /** Método que devuelve el valor index para el anillo 2 de las listas de combinaciones de anillos
	    para este sinograma. Esto es válido cuando está dentro de un sinograma 3d, y hay polar mashing. 
	    @param index índice de las distintas combinaciones de anillos o valores de z que representa cad abin del sinograma.
	    @return valor del anillo 2 para el valor index de las distintas combinaciones polares del sinograma.	
	  */
	  int getRing2FromList(int index){ return ptrListRing2[index];};
	  
	  /** Método que devuelve el valor en z en mm del anillo 1 de las listas de combinaciones de anillos
	    para este sinograma.
	    @param index índice de las distintas combinaciones de anillos o valores de z que representa cad abin del sinograma.
	    @return valor de la posición axial en mm deñ anillo 1 para el valor index de las distintas combinaciones polares del sinograma.	
	  */
	  float getAxialValue1FromList(int index){ return ptrListZ1_mm[index];};
	  
	  /** Método que devuelve el valor en z en mm del anillo 2 de las listas de combinaciones de anillos
	    para este sinograma.
	    @param index índice de las distintas combinaciones de anillos o valores de z que representa cad abin del sinograma.
	    @return valor de la posición axial en mm deñ anillo 2 para el valor index de las distintas combinaciones polares del sinograma.	
	  */
	  float getAxialValue2FromList(int index){ return ptrListZ2_mm[index];};
	  
	  // La vuelvo a definir porque en sinograms2d.. lo necesito y si es protected no puedo.
	  float* getSinogramPtr(){return this->ptrSinogram;};
	  
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
