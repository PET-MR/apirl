/**
	\file GenImagesParameters.h
	\brief Archivo que contiene la definición de la estructura GenImageParameters.

	Este archivo define una estructura con los parámetros necesarios para el comando GenerateImage. Este comando
	recibe por lo general como argumento un archivo *.par donde tiene los valores de los parámetros.
	\todo Terminar de definirlo. Teniendo en cuenta los distintos tipos de shapes que hay, para lo cual habría que agregar una clase
		  por cada uno de ellos.
	\bug
	\warning
	\author Martín Belzunce (martin.sure@gmail.com)
	\date 2010.10.01
	\version 1.0.0
*/

#ifndef _GEN_IMAGE_PARAMETERS_H
#define	_GEN_IMAGE_PARAMETERS_H

#include <Images.h>

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

typedef enum 
{
	SINOGRAMA2D = 0, SINOGRAMA3D = 1, MICHELOGRAMA = 2, MODO_LISTA = 3
}TiposDatosEntrada;

struct DLLEXPORT GenImageParameters
{
	/// Tamaño de la imagen.
	/** El tamaño de la iamgen está definido a través de la estructura SizeImage, donde
		se define la cantidad de píxeles de la misma y el tamaño en mm.
	*/
	SizeImage sizeImage;

}


#endif