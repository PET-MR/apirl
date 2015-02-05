/**
	\file Sinogram3D.h
	\brief Archivo que contiene la definición de la clase abstracta Sinogram3D.

	\todo
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2010.11.11
	\version 1.1.0
*/

#ifndef _SINOGRAM3D_H
#define	_SINOGRAM3D_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <Utilities.h>
#include <Projection.h>
#include <Interfile.h>
#include <SegmentInCylindrical3Dpet.h>
#include <iostream>
#include <sstream>
#include <fstream>

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
/*#ifdef __cplusplus
	extern "C" 
#endif*/ 

/// Clase que define un sinograma de adquisición 3d genérico.
/**	Esta clase define un Sinograma 3D en un formato compatible con el stir. Es decir se guardan todos los
	sinogramas 2D en un mismo archivo ordenados según los segmentos. Cada segmento contiene una cantidad N
	de Sinogramas 2D y permite una diferencia mínima y máxima entre anillos. Por esta razón la clase
	tendrá un array de segmentos, y cada uno de ellos contendrá un array de sinogramas.
*/
class DLLEXPORT Sinogram3D : public Sinogram2D
{
  private:	
	static const unsigned char sizeBin_bytes = sizeof(float);	// Size in bytes of each element of the Michelogram. float -> 4
	/** Función asbtacta en sinogram2d que la implemento para que devuelva siempre false porque no sirve para 3d. */
	bool getPointsFromLor (int indexAng, int indexR, Point2D* p1, Point2D* p2) { return false;};
	/** Función asbtacta en sinogram2d que la implemento para que devuelva siempre false porque no sirve para 3d. */
	bool getPointsFromLor (int indexAng, int indexR, Point2D* p1, Point2D* p2, float* geom) { return false;};
  protected:
	
	/** Número de anillos del sistema. */
	int numRings;	
	
	/** Número de Segmentos que forma el sinograma 3D. */
	int numSegments;
	
	/** Máxima difernecia entre anillos de todo el singorama 3d. */
	int maxRingDiff;
	
	/** Span del sinograma 3d, esto me define la cantidad de segmentos que va a tener el sinograma3d. */
	int span;
	
	/** Puntero a un vector con la cantidad de sinogramas de cada segmento. El vector tiene tantos elementos como segmentos. */
	int* numSinogramsPerSegment;
	
	/** Puntero a un vector con la mínima diferencia de anillo de cada segmento. El vector tiene tantos elementos como segmentos. */
	int* minRingDiffPerSegment;
	
	/** Puntero a un vector con la máxima diferencia de anillo de cada segmento. El vector tiene tantos elementos como segmentos. */
	int* maxRingDiffPerSegment;
	
	/** Radio del fov en mm. */
	//float radioFov_mm;	
	
	/** Largo axial del Field of View en mm, o sea el largo útil del cilindro. */
	float axialFov_mm;
	
	/** Vector con el valor de la coordenada axial (Z) de cada anillo (punto medio del mismo). */
	float* ptrAxialvalues_mm;	 
	  
	// El tipo de dato se debe definir en las clases derivadas. 
	// SegmentInCylindrical3Dpet** segments;	
	/// Función que inicializa los segmentos.
	virtual void inicializarSegmentos () = 0;
	
	/// Setes un sinograma en un segmento.
	virtual void setSinogramInSegment(int indexSegment, int indexSino, Sinogram2DinCylindrical3Dpet* sino2d) = 0;
	
	/** String donde se guardan mensajes del último error ocurrido durante la ejecución. */
	string strError;
  public:
	/// Constructor que crea un sinograma3D nuevo a partir de los parámetros deseados del mismo.
	/** Constructor que crea un sinograma3D nuevo a partir de los parámetros deseados del mismo, estos son
		los tamaños de cada sinograma 2d, los segmentos, el span, la máxima diferencia entre anillos, y los
		tamaños del scanner y del Field of View.
		@param nProj	Número de proyecciones o ángulos de cada sinograma 2d.
		@param nR 	Número de muestras espaciales por proyección del sinograma 2d.
		@param nRings	Número de anillos del scanner.
		@param nSpan	Span del sinograma3d, me define la cantidad de segmentos que va a tener.	
		@param nMaxRingDiff Máxima diferencia entre anillos de este segmento.
		@param rFov_mm	Radio del Field of View en mm.
		@param aFov_mm	Largo axial del Field of View (eje Z) en mm.
		@param rScanner_mm	Radio del scanner en mm.
		
	*/
	Sinogram3D(unsigned int nProj, unsigned int nR, unsigned int nRings, unsigned int nSpan, unsigned int nMaxRingDiff, 
			   float rFov_mm, float aFov_mm);
	
	/// Constructor que solo inicializo lo básico del sinograma3d..
	/** 
		@param	rFov_mm radio del FOV en mm.
		@param	zFov_mm altura axial del FOV en mm.
	*/
	Sinogram3D(float rFov_mm, float zFov_mm);
	
	/// Constructor para cargar los datos de sinogramas a partir del encabezado interfile de un Sinogram3D.
	/** Constructor para cargar los datos de sinogramas a partir del encabezado interfile de un Sinogram3D como está
		definido por el stir. Cono en el interfile no está defnido el tamaño del Fov del scanner ni las dimensiones
		del mismo, se introducen por otro lado dichos parámetros.
		@param	fileHeaderPath	path completo del archivo de header del sinograma 3d en formato interfile. 
		@param	rFov_mm radio del FOV en mm.
		@param	zFov_mm altura axial del FOV en mm.
		@param	rScanner_mm radio del scanner en mm.
	*/
	Sinogram3D(char* fileHeaderPath, float rFov_mm, float zFov_mm);
	
	/// Constructor que genera un nuevo sinograma3d que es un subset de un sinograma orginal.
	/** Genera un sinograma3d cuyo sinogramas 2d son un subsets de los de un sinograma original.
	 * 
	 */
	virtual Sinogram3D* getSubset(int indexSubset, int numSubsets) = 0;
	
	/// Constructor para copia desde otro objeto sinograma3d.
	/** Constructor que inicializa un nuevo objeto Sinogram3D a partir de un objeto ya existente. Hace una copia
		de todos los parámetros.
		@param	srcSinogram3D	sinograma3d que se copiará en esta nueva instancia. 
	*/
	Sinogram3D(Sinogram3D* srcSinogram3D);
	
	/// Destructor.
	/** Es virtual para que cuando se haga un delete de un objeto de una clase derivada, desde un puntero a esta clase
	 * también se llame al destructor de la clase derivada. */
	virtual ~Sinogram3D();
	
	/** Método que deveulve una copia del sinograma3d. Se unsa en vez del constructor en las clases derivadas para mantener abstracción.
		@return puntero a un objeto sinograma3d copia del objetco actual.
	*/
	virtual Sinogram3D* Copy() = 0;
	
	/** Método que devuelve la cantidad de bins totales del sinograma 3D.
	 * @return número de bins que tiene el sinograma 3d.
	 */
	int getBinCount();
	
	/// Método que devuelve la cantidad de sinogramas 2d que tiene el sinograma3d.
	/** Considera cada sinograma 2d almacenado, o sea que las múltiples combinaciones de anillos
	 * para un único bin no las suma.
	 */
	int getNumSinograms();
	
	/** Método que deveulve la cantidad de segmentos del sinograma3d. 
		@return número de segmentos que contiene el sinograma3d.
	*/
	int getNumSegments(){ return numSegments;};
	
	/** Método que deveulve un pùntero al segmento elegido.
		@param indexSegment segmento que se desea obtener.
		@return puntero a objeto del tipo SegmentInCylindrical3Dpet con el segmento pedido.
	*/
	virtual Segment* getSegment(int indexSegment) = 0;
	
	/** Método que deveulve la cantidad de anillos del sinograma3d. 
		@return número de anillos que contiene el sinograma3d.
	*/
	int getNumRings(){ return numRings;};
	
	/** Método que deveulve la máxima diferencia entre anillos del sinograma3d. 
		@return número de máxima diferencia entre anillos del sinograma3d.
	*/
	int getMaxRingDiff(){ return maxRingDiff;};
	
	/** Método que deveulve el largo axial del Field of View del sinograma3d. 
		@return largo axial del field of view en mm.
	*/
	float getAxialFoV_mm(){ return axialFov_mm;};
	
	
	/** Método que devuelve el valor de de la coordenada axial del anillo pedido. */
	float getAxialValue(int indexRing){ return ptrAxialvalues_mm[indexRing];};
	
	/** Método que devuelve un puntero a los valores de las coordenadas de cada anillo.*/
	float* getAxialPtr(){ return ptrAxialvalues_mm;};
	
	/** Método que calcula el likelihood de esta proyección respecto de una de referencia. */
	float getLikelihoodValue(Sinogram3D* referenceProjection);
	
	bool Fill(Event3D* Events, unsigned int NEvents);
	bool FillConstant(float Value);
	// Method that reads the Michelogram data from a file. The dimensions of the
	// expected Michelogram are the ones loaded in the constructor of the class
	bool readFromFile(string fileHeaderPath){};
	bool readFromInterfile(string fileHeaderPath);
	/** Overload to be used in cylindrical scanners in order to set the scanner diameter for each sinogram. */
	bool readFromInterfile(string fileHeaderPath, float radioScanner_mm);
	bool SaveInFile(char* filePath);

	/** Método que escribe el sinograma 2d en formato interfile. 
		El nombre del archivo puede incluir un path.
	*/
	bool writeInterfile(string headerFilename);

	/// Método que realiza la corrección de atenuación, randoms y scatter del sinograma 3d.
	/** Método que realiza la corrección de atenuación, randoms y scatter del sinograma 3d.
	 * La estimación de cada factor debe realizarse por fuera y generar los sinogramas de corrrección, los cuales
	 * son recibidos como parámetro a través del nombre del header del archivo interfile.
	 * Si no se desea aplicar algunos de los factores de corrección se debe pasar string.empty.
		@param acfSinogram sinograma con los attenuation correction factors. Este es un factor multiplicativo.
		@param delayedSinogram sinograma con los eventos de ventana demorada para la corrección de randoms. Este es un factor sustractivo.
		@param scatterSinogram sinograma con la estimación del scatter. Este es un factor sustractivo.
		@return true si pudo realizar la corrección, false en caso contrario (por ejemplo si los tamaños de los sinogramas no coinciden.).
	*/
	bool correctSinogram (string acfSinogram, string delayedSinogram, string scatterSinogram);
	
	/// Método que divide bin a bin con otro sinograma, y lo asigna en el bin de este sinograma.
	/** Método que divide bin a bin con otro sinograma, y lo asigna en el bin de este sinograma. El divisor es el singorama de entrada.
	  @param sinogramDivisor puntero a objeto del tipo Sinogram2D con que devirá bin a bin este sinograma. El divisor con los bines de este singorama.
	  
	  */
	void divideBinToBin(Sinogram3D* sinogramDivisor);
	
	/// Método que multiplica bin a bin con otro sinograma, y lo asigna en el bin de este sinograma.
	/** Método que multiplica bin a bin con otro sinograma, y lo asigna en el bin de este sinograma. 
	  @param sinogramFactor puntero a objeto del tipo Sinogram2D con que multiplicará bin a bin este sinograma. El divisor con los bines de este singorama.
	  
	  */
	void multiplyBinToBin(Sinogram3D* sinogramFactor);
	
	/// Método que divide bin a bin con otro sinograma, y lo asigna en el bin de este sinograma.
	/** Método que divide bin a bin con otro sinograma, y lo asigna en el bin de este sinograma. El divisor es este sinograma.
	  @param sinogramDivisor puntero a objeto del tipo Sinogram2D con que será dividendo de la divsión. 
	  
	  */
	void inverseDivideBinToBin(Sinogram3D* sinogramDividend);
};



#endif