/**
	\file Projection.h
	\brief Archivo que contiene la definición de una clase abstracta Projection.

	Clase abstracta que define una proyección. Es una abstracción para definir clases derivada
	que sean  proyecciones específicas, como Sinogram2D, Michelogram, etc.
	
	
	\todo 
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2010.10.06
	\version 1.1.0
*/

#ifndef _PROJECTION_H
#define _PROJECTION_H

#include <string>

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

class DLLEXPORT Projection
{
  protected:
	  /** String donde se guardan mensajes del último error ocurrido durante la ejecución. */
	  string strError;
  public:
	Projection(){};
	
	/// Método que obtiene un string con el error de algún método luego que devolvió false.
	/** \return objeto del tipos string con el mensaje de error del último error que apareció en el objeto.
	*/
	string getError() {return this->strError;};
		
	virtual float getLikelihoodValue(Projection* referenceProjection){return 0;};
	
	virtual bool readFromInterfile(string headerFilename) = 0;
	
	virtual bool readFromFile(string filename) = 0;
};

#endif // PROJECTION_H