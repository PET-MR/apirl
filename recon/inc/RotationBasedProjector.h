/**
	\file RotationBasedProjector.h
	\brief Archivo que contiene la definición de una clase SiddonProjector.

	Esta clase, es una clase derivada de la clase abstracta Projector. Implementa
	la proyección y retroproyección de distintos tipos de datos utilizando como proyector
	RotationBasedProjector. Este rpoyector funciona píxel wise.
	
	\todo 
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2010.10.06
	\version 1.1.0
*/

#ifndef _ROTATIONBASEDPROJECTOR_H
#define _ROTATIONBASEDPROJECTOR_H

#include <Sinogram2Dtgs.h>
#include <Sinogram3D.h>
#include <Projector.h> 
#include <Images.h>
#include <math.h>
#include <cmath>

using namespace std;

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
  
class DLLEXPORT RotationBasedProjector : virtual Projector
{
  public:
	enum InterpolationMethods{	NEAREST = 0, BILINEAR = 1, BICUBIC = 2 };
  private:
	/// Método de interpolación que se utilizará en el proyector.
	InterpolationMethods interpolationMethod;
	
	/// Imagen binaria que indica que píxeles son parte del fov, y de esta forma si deben ser rotados o no.
	/** Imagen binaria que indica que píxeles son parte del fov, y de esta forma si deben ser rotados o no.
		Cada píxel que vale uno tiene que ser rotado.
	*/
	Image* fovImage;
	
	/// Método privado que realiza la rotación de una imagen.
	/** Método privado que realiza la rotación de una imagen utilizando algunos de los métodos de
		interpolación disponibles. La rotación se realiza teniendo en cuenta un fov circular, de esta
		forma la imagen siempre mantiene el mismo tamaño. La rotación está centrada en el centro del fov.
		@param inputImage puntero a objeto del tipo Image con la imagen a rotar.
		@param rotatedImage puntero a objeto del tipo Image donde se guardará la imagen rotada. Debe estar inicializada con el 
							tamaño de salida deseado.
		@param rotAngle_deg ángulo en grados en el que se desea rotar la imagen.
		@param interpMethod elemento de enumeración del tipo InterpolationMethods que indica el método de interpolación a utilizar en la rotación.
		@return true si la operación fue exitosa.
	*/
	bool RotateImage(Image* inputImage, Image* rotatedImage, float rotAngle_deg, InterpolationMethods interpMethod);
	
  public:
	/** Constructor. */
	RotationBasedProjector(InterpolationMethods intMethod);
	
	/** Constructor. */
	RotationBasedProjector(string intMethod);

	void setInterpolationMethod();
	
	/** Backprojection con RotationBasedProjector para Sinogram2D. */
	bool Backproject (Sinogram2D* InputSinogram, Image* outputImage);  
	/** DivideAndBackprojection con RotationBasedProjector para Sinogram2D. */
	bool DivideAndBackproject (Sinogram2D* InputSinogram, Sinogram2Dtgs* EstimatedSinogram, Image* outputImage);
	/** Projection con RotationBasedProjector para Sinogram2D. */
	bool Project(Image* image, Sinogram2D* projection);
	
	/** Backprojection con RotationBasedProjector para Sinogram3D. */
	bool Backproject (Sinogram3D* InputSinogram, Image* outputImage); 
	/** DivideAndBackprojection con RotationBasedProjector para Sinogram3D. */
	bool DivideAndBackproject (Sinogram3D* InputSinogram, Sinogram3D* EstimatedSinogram, Image* outputImage);
	/** Projection con RotationBasedProjector para Sinogram3D. */
	bool Project(Image* image, Sinogram3D* projection);
};

#endif // PROJECTOR_H