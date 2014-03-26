/**
	\file Sinograms2DmultiSlice.h
	\brief Archivo que contiene la definición de la clase Sinograms2DmultiSlice, que define los sinogramas directos 2d de una adquisición 3d genérica.

	\todo Documentar y adaptar nombre de variables. Dejar la estructura similar a la de Sinogram2D.
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2012.11.11
	\version 1.1.0
*/

#ifndef _SINOGRAMS2DMULTISLICE_H
#define	_SINOGRAMS2DMULTISLICE_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <Utilities.h>
#include <Projection.h>
#include <Sinogram2D.h>
#include <iostream>
#include <fstream>
#include "../medcon/medcon.h"

using namespace::std;
// DLL export/import declaration: visibility of objects
#ifndef LINK_STATIC
	#ifdef WIN32               // Win32 build
		#ifdef DLL_BUILD    // this applies to DLL building
			#define DLLEXPORT __declspec(dllexport)
		#else                   // this applies to DLL clients/users
			#define DLLEXPORT __declspec(dllimport)
		#endif
		#define DLLLOCAL        // not explicitly export-marked objects are local by default on Win32zz
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
/*#ifdef __cplusplus
	extern "C" 
#endif*/ 

/// Clase que define todos los sinogramas directos adquiridos y luego rebinneados de una adquisición 3d.
/**
  Los sinogramas son todos directos y la cantidad depende del número de anillos o de como se adquiera en cristales continuos.
*/
class Sinograms2DmultiSlice : public Sinogram2D
{
  private:
  	virtual bool getPointsFromLor (int indexAng, int indexR, Point2D* p1, Point2D* p2){return false;};
	virtual bool getPointsFromLor (int indexAng, int indexR, Point2D* p1, Point2D* p2, float* geom){return false;};
  protected:
	/** Número de sinogramas 2d totales. */
	int numSinograms;
	/** Largo axial del Field of View en mm, o sea el largo útil del cilindro. */
	float axialFov_mm;	
	/** Vector con el valor de la coordenada axial (Z) de cada anillo (punto medio del mismo). */
	//float* ptrAxialvalues_mm;	
	/** String donde se guardan mensajes del último error ocurrido durante la ejecución. */
	string strError;
	/** Valores axiales de cada sinograma (z). */
	float* ptrAxialValues_mm;
	
	/** Método virtual que inicializa los sinogramas 2d. */
	virtual void initSinograms() = 0;
	
	/** Método que copia los bins de un sinograma que se recibe como parámetro. */
	void CopyBins(Sinograms2DmultiSlice* source);
	
  public:
	/// Constructor que inicializa los sinogramas 2d con todos sus parámetros de dimensiones. 
	/**	Constructor que inicializa los sinogramas 2d con todos sus parámetros de dimensiones. Ellos
		son las dimensiones del sinograma 2d, los tamaños del fov y la cantidad de sinogramas.
		@param nProj	Número de proyecciones o ángulos de cada sinograma 2d.
		@param nR 	Número de muestras espaciales por proyección del sinograma 2d.
		@param rFov_mm	Radio del Field of View en mm.
		@param zFov_mm	Largo Axial del Field of View en mm. Sirve apra saber la separación entre los nSinogramas
		@param nSinograms	Número de sinogramas2d que contiene este segmento. Puede ser NRings o 2*Nrings+1
	*/
	Sinograms2DmultiSlice(int nProj, int nR, float rFov_mm, float zFov_mm, int nSinograms);
	
	/** Constructor que realiza una copia de un set de sinogramas 2d existente.
		@param srcSegment objeto del tipo Sinograms2DinCylindrical3Dpet a partir del cual se creará un nuevo objeto copia de este.
	*/
	Sinograms2DmultiSlice(Sinograms2DmultiSlice* srcSegmento);
	
	/** Constructor que crea y levanta los sinogramas 2d a partir de un archivo interfile. 
	      @param fileName nombre del archivo de entrada.
	      @param rF_mm radio del fov en mm.
	      @param zF_mm largo axial del fov en mm.
	 */
	Sinograms2DmultiSlice(string fileName, float rF_mm, float zF_mm);
	
	/** Destructor. */
	~Sinograms2DmultiSlice();
	
	
	/** Función de inicialización de parámetros, en este caso solo se inicializan los valores de z
	  ya que los parámetros del sinograma como numR o numProj se inicializa en cada sinograma. */
	void initParameters ();

	/** Carga un sinograma a partir de un archivo.
	 * @param fileName nombre del archivo de entrada.
	 * @return true si lo pudo cargar al sinograma, false en caso contrario.
	 */
	bool readFromInterfile(string fileName);
	
	/** Método que escribe el sinograma 2d en formato interfile. 
	 * @param fileName nombre del archivo de salida.
	 * @return true si lo pudo cargar al sinograma, false en caso contrario
	*/
	bool writeInterfile(string headerFilename);
		
	/// Método que devuelve la cantidad de sinogramas totales.
	/**	@return número entero con la cantidad de sinogramas totales.
	*/
	int getNumSinograms() {return numSinograms;};
	
	/** Método que deveulve un puntero al Sinograma2D elegido.
		@param indexSegment sinograma que se desea obtener.
		@return puntero a objeto del tipo Sinogram2DinCylindrical3Dpet con el sinograma pedido.
	*/
	virtual Sinogram2D* getSinogram2D(int indexSinogram2D) = 0;
	
	/** Método que deveulve el largo axial del Field of View del sinograma3d. 
		@return largo axial del field of view en mm.
	*/
	float getAxialFoV_mm(){ return axialFov_mm;};
	
	
	/** Método que devuelve el valor de ángulo de la fila del sinograma pedido. */
	float getAxialValue(int indexRing){ return ptrAxialValues_mm[indexRing];};
	
	/** Método que llena todos los sinogramas con un valor constante */
	bool FillConstant(float Value);
	
	/** Esto es para lograr la abstracción y que queda clase derivada tenga su copia y para  mantener compatibilidad con sinogram2d pero no hace nada.*/
	Sinograms2DmultiSlice* Copy() = 0;
};


#endif
