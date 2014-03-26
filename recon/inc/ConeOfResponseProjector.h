/**
	\file ConeOfResponseProjector.h
	\brief Archivo que contiene la definición de una clase ConeOfResponseProjector.

	Esta clase, es una clase derivada de la clase abstracta Projector. Implementa
	la proyección y retroproyección para Sinogram2Dtgs utilizando como proyector
	el Cono of Response. Este cono de respuesta realiza el Siddon para las distintas
	LORs que pueden ser detectadas por un colimador dado.
	
	\todo Al momento considera que el colimador ideal, después habría que considerar que
		  el plomo va atenuando gradualmente, y que hay cierto fondo.
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2010.11.04
	\version 1.1.0
*/

#ifndef _CONEOFRESPONSEPROJECTOR_H
#define _CONEOFRESPONSEPROJECTOR_H

#include <Sinogram2Dtgs.h>
#include <Projector.h> 
#include <Images.h>

class DLLEXPORT ConeOfResponseProjector : virtual Projector
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
	
  public:
	/** Constructor. */
	ConeOfResponseProjector(int nSamplesOnDetector);
	
	/** Backprojection con Siddon para Sinogram2Dtgs. */
	bool Backproject (Sinogram2Dtgs* InputSinogram, Image* outputImage);  
	/** DivideAndBackprojection con Siddon para Sinogram2Dtgs. */
	bool DivideAndBackproject (Sinogram2Dtgs* InputSinogram, Sinogram2Dtgs* EstimatedSinogram, Image* outputImage);
	/** Projection con Siddon para Sinogram2Dtgs. */
	bool Project(Image* image, Sinogram2Dtgs* projection); 
	
	/** Backprojection con Siddon para Sinogram3D. No es válido para este tipo de proyector. Devuelve false. */
	bool Backproject (Sinogram3D* InputSinogram, Image* outputImage); 
	/** DivideAndBackprojection con Siddon para Sinogram3D. No es válido para este tipo de proyector. Devuelve false. */
	bool DivideAndBackproject (Sinogram3D* InputSinogram, Sinogram3D* EstimatedSinogram, Image* outputImage);
	/** Projection con Siddon para Sinogram3D. No es válido para este tipo de proyector. Devuelve false. */
	bool Project(Image* image, Sinogram3D* projection);
};

#endif // PROJECTOR_H