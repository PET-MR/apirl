/**
	\file Sinogram2D.h
	\brief Archivo que contiene la definición de la clase Sinogram2D.

	Este archivo define la clase Sinogram2. La misma define un sinograma de dos dimensiones genérico
	pensando en PET, o sea que considera que el ángulo de las proyecciones va entre 0 y 180º.
	\todo Extenderlo de manera genérico a distintas geometrías.
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2010.10.01
	\version 1.1.0
*/

#ifndef _SINOGRAM2D_H
#define _SINOGRAM2D_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <string>
#include <Utilities.h>
#include <Projection.h>
#include <Geometry.h>

using namespace std;

/// Clase que define un sinograma de dos dimensiones.
/**
  Clase que define un sinograma de dos dimensiones genérico.
*/
class DLLEXPORT Sinogram2D : public Projection
{
	protected:
	  /** Mínimo ángulo de proyección. */
	  int minAng_deg;
	  /** Máximo ángulo de proyección. */  
	  int maxAng_deg;
	  
	  /** Número de proyecciones o ángulos. */
	  unsigned int numProj;	
	  /** Número de muestras espaciales por proyección. */
	  unsigned int numR;
	  
	  /** Radio del Field of View. */
	  float radioFov_mm;
	  
	  /** Puntero a memoria donde se encuentra el sinograma. Se recorre primero en r,
		  y luego enángulo. Tiene un tamaño de numR*numProj. */
	  float* ptrSinogram;
	  
	  /** Vector con los valores de ángulos en grados de las proyecciones. */
	  float* ptrAngValues_deg;
	  /** Vector con valores de distancia al centro en mm de cada columna del sinograma. */
	  float* ptrRvalues_mm;	  
	  
	  /** String donde se guardan mensajes del último error ocurrido durante la ejecución. */
	  string strError;
	  
	  

	public:
		/** Constructor Base */
		Sinogram2D();

		/** Constructor con rfov */
		Sinogram2D(float rFov_mm);
		
		/** Constructor que recibe como parámetros el tamaño del sinograma*/
		Sinogram2D(unsigned int nProj, unsigned int nR, float rFov_mm);
		
		/** Constructor que realiza una copia de otro objeto Sinogram2D. */
		Sinogram2D(const Sinogram2D* srcSinogram2D);
		
		
		/** Destructor. */
		~Sinogram2D();
		
		/** Función de inicialización de parámetros, se debe llamar cuando se actualizaron algunos
		  de los parámetros del sinograma como numR o numProj. */
		void initParameters ();
	  
		/// Método que obtiene un string con el error de algún método luego que devolvió false.
		/** \return objeto del tipos string con el mensaje de error del último error que apareció en el objeto.
		*/
		string getError() {return this->strError;};
	
		/// Constructor que genera un sinograma subset de otro sinograma dividiendo el original en numSuvest sinogramas.
		/** Se reduce la cantidad de ángulos de las proyecciones numSubsets veces. Los valroes de R quedan igual.
		 * Para esto me quedo solo con los ángulos equiespaciados en numSubsets partiendo desde indexSubsets.
		 * \param srcSinogram2D sinograma 2D con las proyecciones completas del cual se desea obtener un subset.
		 * \param indexSubset indice del subset que se pide.
		 * \param numSubsets numero total de subset.
		 */
		Sinogram2D(const Sinogram2D* srcSinogram2D, int indexSubset, int numSubsets);
		
		/** Método que deveulve una copia del sinograma2d. Se unsa en vez del constructor en las clases derivadas para mantener abstracción.
		@return puntero a un objeto sinograma2d copia del objetco actual.
		*/
		virtual Sinogram2D* Copy() = 0;
	
		/** Método que devuelve cantidad de ángulos de proyecciones. */
		unsigned int getNumProj(){ return numProj;};
		
		/** Método que devuelve cantidad de muestras espaciales de cada proyección. */
		unsigned int getNumR(){ return numR;};
		
		/** Método que devuelve el radio del field of view. */
		float getRadioFov_mm(){ return radioFov_mm;};
		
		/** Método que actualiza y fuerza el radio del field of view a un valor deseado. 
			@param rFov_mm nuevo radio del field of view.
		*/
		void setRadioFov_mm(float rFov_mm);
		
		/** Método que calcula los dos puntos geométricos que forman una Lor. Para el caso de sinograma2d genérico
			se considera que la lor arranca y termina en el fov, mientras que en un singorama2d de un scanner lo hace
			sobre los detectores.
			@param indexAng índice del ángulo del bin del sinograma a procesar.
			@param indexR índice de la distancia r del bin del sinograma a procesar.
			@param p1 puntero a estructura del tipo Point2D donde se guardará el primer punto de la lor (del lado del detector).
			@param p2 puntero a estructura del tipo Point2D donde se guardará el segundo punto de la lor (del otro lado del detector).
			@return devuelve true si encontró los dos puntos sobre el detector, false en caso contrario. 
		*/
		virtual bool getPointsFromLor (int indexAng, int indexR, Point2D* p1, Point2D* p2) = 0;

		/** Método que calcula los dos puntos geométricos que forman una Lor.Y adicionalmente devuelve un peso geométrico, según
		 * las características del scanner.
			@param indexAng índice del ángulo del bin del sinograma a procesar.
			@param indexR índice de la distancia r del bin del sinograma a procesar.
			@param p1 puntero a estructura del tipo Point2D donde se guardará el primer punto de la lor (del lado del detector).
			@param p2 puntero a estructura del tipo Point2D donde se guardará el segundo punto de la lor (del otro lado del detector).
			@param geomFactor factor geométrico calculado a partir de las coordenadas de la lor.
			@return devuelve true si encontró los dos puntos sobre el detector, false en caso contrario. 
		*/
		virtual bool getPointsFromLor (int indexAng, int indexR, Point2D* p1, Point2D* p2, float* geomFactor) = 0;
	
		/** Método que obtiene los dos puntos límites, de entrada y salida, de una lor que cruza el field of view.
			Para esto se le debe pasar las coordenadas del bin del sinograma, y dos estructuras del tipo Point2D
			donde se guardarán los puntos de entrada y salida del fov. Para este caso de sinograma2d genérico
			la lor se obtiene en base al fov del sinograma, en un scanner con una geometría definida, la lor
			se obtendrá en base a dicha geometría.
		    El mismo dependerá del tipo de fov del sinograma. Por default es circular, pero
		    puede ser cuadrado o de otra geometría en clases derivadas.
		  @param indexAng índice del angulo, o sea fila, del bin del sinograma a analizar.
		  @param indexR índice de la distancia R, o sea columna, del bin del sinograma a analizar.
		  @param limitPoint1 estructura del tipo Point2D donde se guardarán las coordenadas de un
				     punto donde corta la recta el fov.
		  @param limitPoint2 estructura del tipo Point2D donde se guardarán las coordenadas de un
				     punto donde corta la recta el fov.
		  @return bool devuelve false si la recta no cruza el FoV y true en el caso contrario.
				Además a través de los parámetros de entrada limitPoint1 y limitPoint2, se devuelven
				las coordenadas del limite del fov con la recta deseada.
		  */
		virtual bool getFovLimits(int indexAng, int indexR, Point2D* limitPoint1, Point2D* limitPoint2);
		
		/** Método que devuelve el valor de un bin del sinograma.
		  @param indexAng índice del angulo, o sea fila, del bin del sinograma a obtener.
		  @param indexR índice de la distancia R, o sea columna, del bin del sinograma a obtener.
		  @return float con el valor del bin (indexAng, indexR) del sinograma.
		  */
		float getSinogramBin(int indexAng, int indexR){ return ptrSinogram[indexAng *numR + indexR];};
		
		/** Método que setea el valor de un bin del sinograma.
		  @param indexAng índice del angulo, o sea fila, del bin del sinograma a escribir.
		  @param indexR índice de la distancia R, o sea columna, del bin del sinograma a escribir.
		  @param value valor que se desea asignar al bin (indexAng, indexR) del sinograma.
		  */
		void setSinogramBin(int indexAng, int indexR, float value){ ptrSinogram[indexAng *numR + indexR] = value;};
		
		/** Método que incrementa en cierta cantridad, el valor de un bin del sinograma.
		  @param indexAng índice del angulo, o sea fila, del bin del sinograma a incrementar.
		  @param indexR índice de la distancia R, o sea columna, del bin del sinograma a incrementar.
		  @param value valor que se desea sumar al bin (indexAng, indexR) del sinograma.
		  */
		void incrementSinogramBin(int indexAng, int indexR, float value){ ptrSinogram[indexAng *numR + indexR] += value;};
		
		/// Método que divide bin a bin con otro sinograma, y lo asigna en el bin de este sinograma.
		/** Método que divide bin a bin con otro sinograma, y lo asigna en el bin de este sinograma. El divisor es el singorama de entrada.
		  @param sinogramDivisor puntero a objeto del tipo Sinogram2D con que devirá bin a bin este sinograma. El divisor con los bines de este singorama.
		  
		  */
		void divideBinToBin(Sinogram2D* sinogramDivisor);
		
		/// Método que divide bin a bin con otro sinograma, y lo asigna en el bin de este sinograma.
		/** Método que divide bin a bin con otro sinograma, y lo asigna en el bin de este sinograma. El divisor es este sinograma.
		  @param sinogramDivisor puntero a objeto del tipo Sinogram2D con que será dividendo de la divsión. 
		  
		  */
		void inverseDivideBinToBin(Sinogram2D* sinogramDividend);
		
		/** Método que devuelve el puntero a memoria del sinograma. */
		float* getSinogramPtr(){ return ptrSinogram;};
		
		/** Método que devuelve el valor de ángulo en grados de la fila del sinograma pedido. */
		float getAngValue(int indexAng){ return ptrAngValues_deg[indexAng];};
		
		/** Método que devuelve un puntero al array con todos los valroes de ángulo */
		float* getAngPtr(){ return ptrAngValues_deg;};
		
		/** Método que devuelve el valor de ángulo de la fila del sinograma pedido. */
		float getRValue(int indexR){ return ptrRvalues_mm[indexR];};
		
		/** Método que devuelve un puntero al array con todos los valroes de ángulo. */
		float getRPtr(){ return ptrRvalues_mm;};
		
		/** Método que calcula el likelihood de esta proyección respecto de una de referencia. */
		float getLikelihoodValue(Sinogram2D* referenceProjection);
		
		/** Método que lee el sinograma 2d en formato interfile. */
		bool readFromInterfile(string headerFilename);
		
		/** Método que escribe el sinograma 2d en formato interfile. 
			El nombre del archivo puede incluir un path.
		*/
		bool writeInterfile(string headerFilename);
		
		bool readFromFile(string filePath);
		
		bool Fill(Event2D* Events, unsigned int NEvents);
		
		virtual bool FillConstant(float Value);

		bool SaveInFile(char* path);


		bool ReadFromArray(float* SinogramArray);
		
		float* getAnglesInRadians();
};

#endif
