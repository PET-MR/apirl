/**
	\file SystemMatrix.h
	\brief Archivo que contiene la definición de la clase SystemMatrix.

	Este archivo define la clase SystemMatrix, que contiene una matriz de sistema usada en la
	reconstrucción. Una matriz de sistema es la que permite obtener las proyecciones a partir
	de una imagen. Por lo que P = A*X. P es un vector que contiene todos los elementos del 
	sinograma o proyección, X un vector con todos los píxeles de la imagen, y A la matriz del
	sistema que tiene tantas columnas como píxeles de la imagen, y tantas filas como elementos
	las proyeccion. Básicamente es una matriz.
	\todo 
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2010.10.01
	\version 1.0.0
*/

#ifndef _SYSTEMMATRIX_H
#define _SYSTEMMATRIX_H

#include <Image.h>
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

class DLLEXPORT SystemMatrix
{
  private:
	/** Cantidad de filas (bins de las proyecciones) de la matriz. */
	int numRows;
	/** Cantidad de columnas (píxeles de la imagen) de la matriz. */
	int numCols;
	
	/** Puntero a la matriz. Los elementos están guardados en orden fila - columna. */
	float* ptrSystemMatrix;
	
	/** Cantidad de elementos de la matriz. */
	int numBins;
	/** Cantidad de elementos distintos de cero en la matriz. */
	int numNonZeroBins;
	
  public:
	/** Constructor . */
	SystemMatrix(image);
	

};

#endif
