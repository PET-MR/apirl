/**
	\file ConeOfResponseWithPenetrationProjectorWithAttenuation.h
	\brief Archivo que contiene la definición de una clase ConeOfResponseWithPenetrationProjectorWithAttenuation.

	Esta clase, es una clase derivada de la clase abstracta Projector. Implementa
	la proyección y retroproyección para Sinogram2Dtgs utilizando como proyector
	el Cono of Response pero teniendo en cuenta también el fondo que se mete a través
	del plomo. Este cono de respuesta realiza el Siddon para las distintas
	LORs que pueden ser detectadas por un colimador dado. Las que solo pasan por aire, o sea
	aquellas que se consideran en el projector ConeOfResponse, se mantienen como en dicho proyector,
	pero ahora también se consideran más LORs, las que pasan parcialmente por el plomo y llegan
	a la cara del detector. A estas LORs se le aplica el coeficiente de atenuación exp(-mu*l) en los
	pesos de siddon, siendo l el largo en el plomo, mu el coeficiente de atenuación lineal del plomo
	para esa energía.
	La cantidad de lineas que se procesan depende de dos parámetros: numSamplesOnDetector y 
	numSamplesOnCollimatorSurf. Que son la cantidad de puntos que se tienen en cuenta sobre el detector,
	Además realiza corrección por atenuación.
	
	\todo Por ahora el colimador solo puede ser de plomo, y se considera la energía del cesio. Después debería ser parámetro.
		  También el detector se considera de 2 pulgadas. Esto habría que incorporarlo a sinograma2dtgs tal vez, aunque
		  no pareciera que fuera a cambiar.
		  Inicializar todos los segmentos que atraviezan el colimador en el constructor, ya que no dependen del bin, 
		  sino de la cantidad de muestras y de la configuración del colimador. Al ahcer esto no hay que recalcularlo
		  para cada lor de cada bin, o sea se divide el tiempo de procesamiento de esta etapa por el número de bins.
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2010.11.23
	\version 1.1.0
*/

#ifndef _CONEOFRESPONSEWITHPENETRATIONPROJECTORWITHATTENUATION_H
#define _CONEOFRESPONSEWITHPENETRATIONPROJECTORWITHATTENUATION_H

#include <Sinogram2Dtgs.h>
#include <Projector.h> 
#include <Images.h>

class DLLEXPORT ConeOfResponseWithPenetrationProjectorWithAttenuation : virtual Projector
{
  private:
	/// Número de muestras sobre el detector para generar el cono de respuesta.
	/** Número de muestras sobre el detector para generar el cono de respuesta. Las mismás
		se consideran sobre la superficie visibles del detector. En combinación con los puntos
		sobre la cara exterior del colimador me dará la cantidad de líneas de Siddon utilizadas en el cono de respuesta.
		*/
	int numSamplesOnDetector;
	
	/// Número de muestras sobre la cara externa del colimador para generar el cono de respuesta.
	/** Número de muestras sobre la cara externa del colimador para generar el cono de respuesta. Las mismás
		se consideran sobre la superficie total del colimador del detector (no solo el agujero). En combinación con los puntos
		sobre el detector me dará la cantidad de líneas de Siddon utilizadas en el cono de respuesta.
		*/
	int numSamplesOnCollimatorSurf;
	
	/** Número de líneas evaluadas para el cone of response. */
	int numLinesPerCone;
	
	/** Coeficiente de atenuación lineal del colimador. Depende del material y la energía de trabajo.
		Se recibe como parámetro. Está en 1/cm.*/
	float linearAttenuationCoeficcient_cm;
	
	/// Umbral de atenuación a partir del cual no se procesa la LOR analizadas.
	/** Umbral de atenuación a partir del cual no se procesa la LOR analizadas. Al momento lo fijamos
		en 0.95 (o sea cuando solo cruzan el 5%). Este umbral se calcula realizando exp(-mu*l), siendo
		mu el coeficiente de atenuación lineal y l el largo de la trayectoría a través del material
		del colimador. */
	const static float attenuationThreshold = 0.95;
	
	/// Umbral del largo de una línea que cruza el colimador a partir del cual no se la prcoesa.
	/* Este largo es el máximo largo de una línea que atravieza el material del colimador, para que 
		sea procesada. A partir de ese largo no se procesa. Se calcula a partir del attenuationThreshold
		y el linearAttenuationCoeficcient recibido como parámetro en el constructor. */
	float lengthInCollimatorThreshold_mm;
	
	/** Método que obtiene el coeficiente de atenuación para un largo de segmento en el colimador.
		@param lengthSegment_mm distancia que recorre una lor dentro del colimador.
		@return float proporción de fotones atenuados dentro del colimador.
	*/
	float getAttenuationWeight(float lengthSegment_mm);
	
	/// Imagen con mapa de atenuación con coeficientes de atenuación lineal en 1/mm.
	Image* attenuationMap;
	
  public:
	/** Constructor. */
	ConeOfResponseWithPenetrationProjectorWithAttenuation(int nSamplesOnDetector, int nSamplesOnCollimatorSurf, float linAttCoef, Image* attenuationMap);
	
	/** Backprojection con CoRwPwA para Sinogram2Dtgs. */
	bool Backproject (Sinogram2Dtgs* InputSinogram, Image* outputImage);  
	/** Projection con CoRwPwA para Sinogram2Dtgs. */
	bool Project(Image* image, Sinogram2Dtgs* projection);
};

#endif // PROJECTOR_H