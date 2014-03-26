/**
	\file ArPetProjector.h
	\brief Archivo que contiene la definición de una clase ArPetProjector.

	Esta clase, es una clase derivada de la clase abstracta Projector. Implementa
	la proyección y retroproyección de sinograma3d y 2d del ar-pet. Tiene en cuenta seu geometría hexagonal
	y como esto afecta a la sensibilidad de cada lor.
	
	\todo 
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2013.04.24
	\version 1.1.0
*/

#ifndef _ARPETPROJECTOR_H
#define _ARPETPROJECTOR_H

#include <Sinogram2Dtgs.h>
#include <Sinogram3D.h>
#include <Sinogram2Din3DArPet.h>
#include <Projector.h> 
#include <Images.h>

class DLLEXPORT ArPetProjector : virtual Projector
{
  private:
	/// Número de lors utilizadas en el siddon por bin del sinograma.
	/** Número de lors utilizadas en el siddon por bin del sinograma. Por default es una lor sola
		que sale del centro del detector. Si se configuraran más son n líneas paralelas equiespaciadas
		sobre el detector por cada bin del sinograma.
		*/
	int numSamplesOnDetector;
  public:
	/// Constructor base. 
	/** El cosntructor base setea una lor por bin. */
	ArPetProjector();
	/** Este constructor setea la cantidad de lors por bin que se desea utilizar. */
	ArPetProjector(int nSamplesOnDetector);
	
	/** Backprojection con ModeloArPet para Sinogram2D. */
	bool Backproject (Sinogram2D* InputSinogram, Image* outputImage);  
	/** DivideAndBackprojection con ModeloArPet para Sinogram2D. */
	bool DivideAndBackproject (Sinogram2D* InputSinogram, Sinogram2D* EstimatedSinogram, Image* outputImage);
	/** Projection con ModeloArPet para Sinogram2D. */
	bool Project(Image* image, Sinogram2D* projection);
	
	// Antes iba a hacer un proyector especial para el tipo de dato, pero por ahora no. Ya que encapsule la parte geomerica en el sinograma
	// y uso todo como si fuera Sinogram3D.
// 	/** Backprojection con ModeloArPet para Sinogram2Din3DArPet. */
// 	bool Backproject (Sinogram2Din3DArPet* InputSinogram, Image* outputImage);  
// 	/** DivideAndBackprojection con ModeloArPet para Sinogram2Din3DArPet. */
// 	bool DivideAndBackproject (Sinogram2Din3DArPet* InputSinogram, Sinogram2Din3DArPet* EstimatedSinogram, Image* outputImage);
// 	/** Projection con ModeloArPet para Sinogram2Din3DArPet. */
// 	bool Project(Image* image, Sinogram2Din3DArPet* projection);
	
	/** Backprojection con ModeloArPet para Sinogram3D. */
	bool Backproject (Sinogram3D* InputSinogram, Image* outputImage); 
	/** DivideAndBackprojection con ModeloArPet para Sinogram3D. */
	bool DivideAndBackproject (Sinogram3D* InputSinogram, Sinogram3D* EstimatedSinogram, Image* outputImage);
	/** Projection con ModeloArPet para Sinogram3D. */
	bool Project(Image* image, Sinogram3D* projection);
};

#endif // PROJECTOR_H