
/**
	\file Projector.h
	\brief Archivo que contiene la definición de una clase abstracta Projector.

	A partir de esta clase abstracta se definen distintos proyectores. La idea de hacerlo así
	es que MLEM pueda llamar a través de Projector distintos proyectores de la misma manera.
	
	\todo 
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2010.10.06
	\version 1.1.0
*/

#include <Projection.h>
#include <Sinogram2Dtgs.h>
#include <Sinogram3D.h>
#include <Images.h>

#ifndef _PROJECTOR_H
#define _PROJECTOR_H

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
/** Clase abstracta Projector. */
class DLLEXPORT Projector
{
  protected:
    /// Flag que indica si se desea usar todas las combinaciones axiales.
    /** Flag que indica si se desea usar todas las combinaciones axial para un bin. Esto es válido solo para
      * proyectores en 3d. */
    bool useMultipleLorsPerBin;
  public:
    /** Constructor. */
    Projector(){ useMultipleLorsPerBin = 1;};
    
    /** Método que setea el flag useMultipleLorsPerBin. */
    void setMultipleLorsPerBin(bool enable) {useMultipleLorsPerBin = enable;};
    
    /** Método que obtiene el valor del flag useMultipleLorsPerBin. */
    bool getMultipleLorsPerBin() {return useMultipleLorsPerBin;};
    
    /** Método abstracto de Project para Sinogram2d. */
    virtual bool Project(Image*,Sinogram2D*){};
    /** Método abstracto de Backroject para Sinogram2d. */
    virtual bool Backproject(Sinogram2D*, Image*){};
    /** Método abstracto de DivideAndBackproject para Sinogram2d. */
    virtual bool DivideAndBackproject (Sinogram2D* InputSinogram, Sinogram2D* EstimatedSinogram, Image* outputImage){};
	
    /** Método abstracto de Project para Sinogram2dTgs. */
    virtual bool Project(Image*,Sinogram2Dtgs*){};
    /** Método abstracto de Backroject para Sinogram2dTgs. */
    virtual bool Backproject(Sinogram2Dtgs*, Image*){};
    /** Método abstracto de DivideAndBackproject para Sinogram2dTgs. */
    virtual bool DivideAndBackproject (Sinogram2Dtgs* InputSinogram, Sinogram2Dtgs* EstimatedSinogram, Image* outputImage){};
    
    /** Método abstracto de Project para Sinogram3D. */
    virtual bool Project(Image*,Sinogram3D*){};
    /** Método abstracto de Backroject para Sinogram3D. */
    virtual bool Backproject(Sinogram3D*, Image*){};
    /** Método abstracto de DivideAndBackproject para Sinogram3D. */
    virtual bool DivideAndBackproject (Sinogram3D* InputSinogram, Sinogram3D* EstimatedSinogram, Image* outputImage) {};
};

#endif // PROJECTOR_H
