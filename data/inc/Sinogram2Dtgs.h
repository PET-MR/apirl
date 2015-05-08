/**
	\file Sinogram2Dtgs.h
	\brief Archivo que contiene la definición de la clase Sinogram2Dtgs, clase derivada de Sinogram2D.

	Este archivo define la clase Sinogram2Dtgs, que es una clase derivada de Sinogram2D.
	Al ser un sinograma de SPECT tiene proyecciones de 0 a 360º. Además le agrego propiedades
	que hacen a este tipo de sinograma, por ejemplo el largo y ancho del colimador, para poder
	obtener el Cone of Response para cada lor.
	\todo 
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2010.10.01
	\version 1.0.0
*/

#ifndef _SINOGRAM2DTGS_H
#define _SINOGRAM2DTGS_H

#include <Sinogram2D.h>

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

class DLLEXPORT Sinogram2Dtgs : public Sinogram2D
{
  protected:
	/** Largo del colimador en mm. */
	float lengthColimator_mm;
	/** Diámetro del agujero colimador en mm. */
	float widthHoleCollimator_mm;
	/** Diámetro total del colimador en mm. */
	float widthCollimator_mm;
	/** Distancias del centro del field of view al borde del cristal. */
	float distCrystalToCenterFov;
	/// Diámetro de detector. 
	/** Diámetro del detector. Es el total, sin importar cuanto está descubierto por el agujero del colimador. 
		Para eso está la variables widthHoleCollimator_mm. */
	const static float widthDetector_mm = 50.8;
	
  public:
	/** Constructor que solo inicializa minAng y maxAng. */
	Sinogram2Dtgs();
	
	/** Constructor para Sinogram2Dtgs. */
	Sinogram2Dtgs(unsigned int numProj, unsigned int numR, float rFov_mm, float distCrystalToCenterFov, 
				float lengthColimator_mm, float widthCollimator_mm, float widthHoleCollimator_mm);

	/// Constructor de Copia
	Sinogram2Dtgs(const Sinogram2Dtgs* srcSinogram2Dtgs);
	
	/** Método que deveulve una copia del sinograma2d. Se unsa en vez del constructor en las clases derivadas para mantener abstracción.
		@return puntero a un objeto sinograma2d copia del objetco actual.
	*/
	Sinogram2D* Copy(){ Sinogram2Dtgs* sino2dcopy = new Sinogram2Dtgs(this); return (Sinogram2D*)sino2dcopy;	};
	
	
	using Sinogram2D::getFovLimits; // To avoid the warning on possible unintended override.
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
	
	/** Método que obtiene el punto medio sobre la superficie del detector, para un bin dado del sinograma.
		@param indexAng índice del ángulo del bin del sinograma a procesar.
		@param indexR índice de la distancia r del bin del sinograma a procesar.
		@return Point2D estructura con las coordenadas del punto sobre la cara del detector.
	*/
	Point2D getPointOnDetector(int indexAng, int indexR);
	
	/** Método que obtiene un punto, desplazado un offset del centro, sobre la superficie del detector, 
		para un bin dado del sinograma. 
		@param indexAng índice del ángulo del bin del sinograma a procesar.
		@param indexR índice de la distancia r del bin del sinograma a procesar.
		@param offsetDetector_mm distancia desde el centro del detector al punto deseado, ambos sobre la superficie del detector. 
		@return Point2D estructura con las coordenadas del punto sobre la cara del detector.
	*/
	Point2D getPointOnDetector(int indexAng, int indexR, float offsetDetector_mm);
	
	/** Método que obtiene un punto, desplazado un offset del centro, sobre la superficie del colimador, 
		para un bin dado del sinograma. 
		@param indexAng índice del ángulo del bin del sinograma a procesar.
		@param indexR índice de la distancia r del bin del sinograma a procesar.
		@param offsetCollimator_mm distancia desde el centro del colimador al punto deseado, ambos sobre la superficie del colimador.
		@return Point2D estructura con las coordenadas del punto sobre la cara del detector.
	*/
	Point2D getPointOnCollimatorSurface(int indexAng, int indexR, float offsetCollimator_mm);
	
	/** Método que calcula los dos puntos que forman una Lor. Solo sirve cuando se usa ún único rayo por bin.
		@param indexAng índice del ángulo del bin del sinograma a procesar.
		@param indexR índice de la distancia r del bin del sinograma a procesar.
		@param	p1 puntero a estructura del tipo Point2D donde se guardará el primer punto de la lor (del lado del detector).
		@param	p2 puntero a estructura del tipo Point2D donde se guardará el segundo punto de la lor (del otro lado del detector).
	*/
	bool getPointsFromLor (int indexAng, int indexR, Point2D* p1, Point2D* p2){getPointsFromTgsLor (indexAng, indexR, 0, 0, p1, p2);return true;};
	
	bool getPointsFromLor (int indexAng, int indexR, Point2D* p1, Point2D* p2, float* geom) { getPointsFromTgsLor (indexAng, indexR, 0, 0, p1, p2); return true;};
	
	/** Método que calcula los dos puntos geométricos que forman una Lor sobremuestrada.Y adicionalmente devuelve un peso geométrico, según
	  * las características del scanner. Para sobremuestrear la LOR, se le indica en cuantos puntos se divide cada LOR y cual de las muestras
	  * se desea. O sea que para N submuestras, se puede solicitar índices entre 0 y N-1. Serviría para una adquisición con desplazamiento
	  * continuo.
		@param indexAng índice del ángulo del bin del sinograma a procesar.
		@param indexR índice de la distancia r del bin del sinograma a procesar.
		@param indexSubsample índice de la submuestra para la LOR (indexAng,indexR).
		@param numSubsamples número de submuestras por LOR.
		@param p1 puntero a estructura del tipo Point2D donde se guardará el primer punto de la lor (del lado del detector).
		@param p2 puntero a estructura del tipo Point2D donde se guardará el segundo punto de la lor (del otro lado del detector).
		@param geomFactor factor geométrico calculado a partir de las coordenadas de la lor.
		@return devuelve true si encontró los dos puntos sobre el detector, false en caso contrario. 
	*/
	bool getPointsFromOverSampledLor (int indexAng, int indexR, int indexSubsample, int numSubsamples, Point2D* p1, Point2D* p2, float* geomFactor);
	
	/** Método que calcula los dos puntos que forman una Lor. El primer punto se obtiene en el centro del colimador, y
		el otro punto en la cara opuesta a la misma distancia del centro del FoV. La lor es la de un punto del bin, con un
		offset sobre el detector y sobre la cara del colimador. Por lo que hay muchas lors por cada bin del sinograma, 
		con distintos ángulos de entrada. Para obtener la lor "ideal", los offsets deberían ser 0.
		Los puntos que definen la lor son los definidos por los offsets, pero los puntos devueltos son los de los extremos de 
		la lor, esto es uno sobre el punto medio del colimador, y otro en el lado opuesto.
		@param indexAng índice del ángulo del bin del sinograma a procesar.
		@param indexR índice de la distancia r del bin del sinograma a procesar.
		@param offsetDetector_mm distancia desde el centro del detector, del primer punto de la lor sobre la superifice del mismo. 
		@param offsetCollimatorSurface_mm distnacia desde el centro del colimador del segundo punto que define la lor, y se encuentra
				sobre la superficie del mismo.
		@param	p1 puntero a estructura del tipo Point2D donde se guardará el primer punto de la lor (del lado del detector).
		@param	p2 puntero a estructura del tipo Point2D donde se guardará el segundo punto de la lor (del otro lado del detector).
	*/
	void getPointsFromTgsLor (int indexAng, int indexR, float offsetDetector_mm, float offsetCollimator_mm, Point2D* p1, Point2D* p2);

	/**	Método que obtiene el largo del segmento resultante de una lor con el material del colimador.
		O sea, me devuelve la distancia que recorre el fotón gamma por el material del colimador. Sirve
		para saber cuanto se atenúa una lor en particular. Para todos los bins del sinograma es lo mismo
		ya que solo depende de la geometría del colimador, y de una lor relativa al mismo.
		Por lo que se necesita las posiciones sobre el detector y la superficie del colimador que forman la lor a analizar
		y los parámetros	del colimador.
		Tiene en cuenta que la lor puede pasar por plomo-agujero-plomo.
		@param offsetDetector_mm distancia desde el centro del detector, del primer punto de la lor sobre la superifice del mismo. 
		@param offsetCollimatorSurface_mm distnacia desde el centro del colimador del segundo punto que forma la lor, y se encuentra sobre la superficie del mismo.
		@return float distancia de la lor que atravieza el material del colimador. Devuelve 0 si no atravieza material (solo pasa por el agujero).
	*/
	float getSegmentLengthInCollimator(float offsetDetector_mm, float offsetCollimatorSurface_mm);
	
	/** Función que devuelve el largo del colimador utilizado para este sinograma. */
	float getLengthColimator_mm() {return lengthColimator_mm;};
	
	/** Función que devuelve el diámetro del colimador utilizado para este sinograma. */
	float getWidthCollimator_mm() {return widthCollimator_mm;};
	
	/** Función que devuelve el diámetro del agujero del colimador utilizado para este sinograma. */
	float getWidthHoleCollimator_mm() {return widthHoleCollimator_mm;};
	
	/** Función que devuelve el diámetro del colimador utilizado para este sinograma. */
	float getWidthDetector_mm() {return widthDetector_mm;};
	
	/** Función que devuelve la distancia del frente del cristal al centro del fov. */
	float getDistCrystalToCenterFov() {return distCrystalToCenterFov;};
	
	/** Función que setes los parámetros geométricos del sinograma adquirido. */
	void setGeometricParameters(float rFov_mm, float dCrystalToCenterFov, float lColimator_mm, float wCollimator_mm, float wHoleCollimator_mm);
};

#endif
