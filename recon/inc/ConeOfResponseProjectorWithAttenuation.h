/**
	\file ConeOfResponseProjectorWithAttenuation.h
	\brief Archivo que contiene la definición de una clase ConeOfResponseProjectorWithAttenuation.

	Esta clase, es una clase derivada de la clase abstracta Projector. Implementa
	la proyección y retroproyección para Sinogram2Dtgs utilizando como proyector
	el Cono of Response con atenuación. Este cono de respuesta realiza el Siddon para las distintas
	LORs que pueden ser detectadas por un colimador dado y la atenuación sufrida
	por cada una de ellas.
	
	\todo Al momento considera que el colimador ideal, después habría que considerar que
		  el plomo va atenuando gradualmente, y que hay cierto fondo.
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2010.11.04
	\version 1.1.0
*/

#ifndef _CONEOFRESPONSEPROJECTORWITHATTENUATION_H
#define _CONEOFRESPONSEPROJECTORWITHATTENUATION_H

#include <Sinogram2Dtgs.h>
#include <Sinogram2DtgsInSegment.h>
#include <Projector.h> 
#include <Images.h>
#include <math.h>
#include <Geometry.h>

class DLLEXPORT ConeOfResponseProjectorWithAttenuation : virtual Projector
{
  private:
	/// Número de muestras sobre el detector para generar el cono de respuesta.
	/** Número de muestras sobre el detector para generar el cono de respuesta. Las mismás
		se consideran sobre la superficie visibles del detector. Las mismas posiciones son
		consideradas para el agujero del colimador sobre su cara externa. La combinación de
		dichos puntos me dará la cantidad de líneas de Siddon utilizadas en el cono de respuesta.
		La cantidad de lineas es numSamplesOnDetector^2. */
	int numSamplesOnDetector;
	
	/** Número de líneas evaluadas para el cone of response. */
	int numLinesPerCone;
	
	/// Imagen donde se almacenan factores de corrección por la apertura en el eje z del colimador. 
	/** Imagen donde se almacenan factores de corrección por la apertura en el eje z del colimador.
		Esto es necesario para reconstrucción 2d donde el espesor del segmento no es despreciable, o
		sea para el tipo de dato Sinogram2DtgsInSegment.
	*/
	Image* sensitivityCorrectionImageForSegmentWidth;
	
	/// Imagen con mapa de atenuación con coeficientes de atenuación lineal en 1/mm.
	Image* attenuationMap;
	
	void ComputeCorrectionImageForSegmentWidth(SizeImage size, Sinogram2DtgsInSegment* InputSinogram);
	
	float ComputeIntersectedSegmentForPixel(float distPixelYfrenteDetector_mm,  float distCrystalToCenterFov_mm, float lengthColimator_mm, float widthHoleCollimator_mm, float widthSegment_mm);
	float ComputeIntersectedSegmentForPixel(float distPixelXCentroCol_mm, float distPixelYfrenteDetector_mm,  float distCrystalToCenterFov_mm, float lengthColimator_mm, float widthHoleCollimator_mm, float widthSegment_mm);
  public:
	/** Constructor. */
	ConeOfResponseProjectorWithAttenuation(int nSamplesOnDetector, Image* attenuationMap);
	
	/// Backrojection con ConeOfResponse para Sinogram2Dtgs que tiene en cuenta la atenuación del medio
	/** Realiza la backprojection del cociente de dos sinogramas teniendo en cuenta la atenuación del medio. 
		O sea, que para este proyector no hay que aplicar una corrección por atenuación con factores
		multiplicativos al sinograma de entrada, o al sinograma proyectado según como se desee.
	*/
	bool Backproject (Sinogram2Dtgs* InputSinogram, Image* outputImage);  

	/// Projection con ConeOfResponse para Sinogram2Dtgs que tiene en cuenta la atenuación del medio
	/** Projection con ConeOfResponse para Sinogram2Dtgs que tiene en cuenta la atenuación del medio. 
		O sea, que para este proyector no hay que aplicar una corrección por atenuación con factores
		multiplicativos al sinograma de entrada, o al sinograma proyectado según como se desee.
	*/
	bool Project(Image* image, Sinogram2Dtgs* projection); 

};

#endif // PROJECTOR_H