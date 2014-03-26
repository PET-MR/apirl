/**
	\file Sinogram2Dtgs.h
	\brief Archivo que contiene la definición de la clase Sinogram2DtgsInSegment, clase derivada de Sinogram2Dtgs.

	Este archivo define la clase Sinogram2DtgsInSegment, que es una clase derivada de Sinogram2Dtgs.
	Sinogram2Dtgs es un sinograma 2d para tgs que considera el slice medido sin espesor. O sea que
	la actividad se distribuye en un único plano 2D. Sinogram2DtgsInSegment es una extensión de dicho
	sinograma 2d a un slice pero que su espesor no es despreciable, sino que está dado por un segmento 
	(slices del tambor) de cierto espesor. Esto hace que tenga que tenerse en cuenta la sensibilidad del
	colimador en el eje z (altura) también. Ya que la sensibilidad de cada punto del slice irá cambiando.
	Esto es importante en la reconstrucción, ya que por más que se desee reconstruir un solo slice, o sea
	una imagen 2D, al hacerse la adquisición sobre un segmento el modelo geométrico cambia mucho, y sino 
	se lo tiene en cuenta al reconstrucción ser errónea.
	Básicamente esta clase lo único que agrega respecto de Sinogram2Dtgs es el espesor del segmento adquirido.
	O sea que por más que el sinograma siga siendo bidimensional, ahora tiene como propiedad geométrica también
	el espesor del segmento.
	
	\todo 
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2012.01.06
	\version 1.1.0
*/

#ifndef _SINOGRAM2DTGSINSEGMENT_H
#define _SINOGRAM2DTGSINSEGMENT_H

#include <Sinogram2Dtgs.h>

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

class DLLEXPORT Sinogram2DtgsInSegment : public Sinogram2Dtgs
{
  private:
	/** Altura del segmento en mm. */
	float widthSegment_mm;
	
  public:
	/** Constructor que solo inicializa minAng y maxAng. */
	Sinogram2DtgsInSegment();
	
	/** Constructor para Sinogram2Dtgs. */
	Sinogram2DtgsInSegment(unsigned int numProj, unsigned int numR, float rFov_mm, float distCrystalToCenterFov, 
				float lengthColimator_mm, float widthCollimator_mm, float widthHoleCollimator_mm, float wSegment_mm);

	/// Constructor de Copia
	Sinogram2DtgsInSegment(const Sinogram2DtgsInSegment* srcSinogram2Dtgs);
	
	/** Función que devuelve el ancho del segmento utilizado para este sinograma. */
	float getWidthSegment_mm() {return widthSegment_mm;};
	
	/** Función que setes los parámetros geométricos del sinograma adquirido. */
	void setGeometricParameters(float rFov_mm, float dCrystalToCenterFov, float lColimator_mm, float wCollimator_mm, float wHoleCollimator_mm, float wSegment_mm);
};

#endif
