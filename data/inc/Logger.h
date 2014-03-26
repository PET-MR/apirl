/**
	\file Logger.h
	\brief Archivo que contiene la definici�n de la clase Logger.

	Agregar m�s detalles.
	\todo
	\bug
	\warning
	\author Martn Belzunce (martin.sure@gmail.com)
	\date 2010.12.06
	\version 1.0.0
*/

#ifndef _LOGGER_H
#define	_LOGGER_H

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

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string.h>
#include <string>
using namespace std;

/**
	\brief Clase que maneja un loggeador de datos o eventos en un archivo.

	Mediante esta clase se maneja un registrador de strings en un archivo de texto. De esta
	forma se lo puede utilziar para registrar eventos, con fecha y hora; simplemente datos
	que se deseen registrar en un archivo de texto; e incluso n�meros separados por coma para
	ser importados desde otra aplicaci�n.

	\todo Agregar templat para la funci�n write value, de manera que value pueda ser cualquier tipo de dato. Idem para writeRowOfNumbers.
	\todo Agregar la posibilidad de que el caracter de fin de l�nea pueda ser seleccionable.
	\todo Agregar strError y getError para el manejo de errores.
*/
class DLLEXPORT Logger
{
public:
  	/// Constructor de la clase Logger.
	/** Este constructor instancia la clase Logger, y crea el archivo de texto
		que se utilizar�. PAra esto recibe como par�metro el nombre del archivo 
		completo.
	*/
	Logger(string fullFileName);
	
	/// Destructor de la clase.
	/** Destructor de la clase, verifica que est� cerrado el archivo, y si no lo
		est� lo cierra.
	*/
	~Logger();

	/// Escribe un string en una l�nea del archivo de texto.
	/** Escribe una l�nea en el archivo de texto, para esto recibe un string con
		la l�nea a escribir, y luego se le agrega el caracter de fin de l�nea. 
		Utilizamos la codificaci�n de Linux, para el caracter de fin de l�nea(LN).
		\param String con la l�nea a escribir, no debe incluir el caracter de fin de l�nea.
		\return True si se pudo realizar la escritura, false en caso contrario.
	*/
	bool writeLine(string line);
	
		/// Escribe un string en una l�nea del archivo de texto.
	/** Sobrecarga del m�todo writeLine para strings del tipo c (char*).
		\param Char con un c string con la l�nea a escribir, no debe incluir el caracter de fin de l�nea.
		\param Int con el largo del string de c.
		\return True si se pudo realizar la escritura, false en caso contrario.
	*/
	bool writeLine(char* line, int length);
	
	/// Escribe el registro de un valor con el formato Label: Value.
	/** Escribe el registro de un valor identificado  con una etiqueta, para
		esto se recibe como par�metros la etiqueta, y el valor que toma. Los mismos
		se escriben con el formato "label: value", y se agrega el caracter de fin de l�nea.
		\param String con la etiqueta(label) con que se identificar� el valor a escribir.
		\param Valor a escribir.
		\return True si se pudo realizar la escritura, false en caso contrario.
	*/
	bool writeValue (string label, string value);
	
	/// Escribe el registro de un valor con el formato Label: Value.
	/** Escribe el registro de un valor identificado  con una etiqueta, para
		esto se recibe como par�metros la etiqueta, y el valor que toma. Los mismos
		se escriben con el formato "label: value", y se agrega el caracter de fin de l�nea.
		\param Char* con un C string  con la etiqueta(label) que identificar� el valor a escribir.
		\param Int con el largo del string label.
		\param Char* con un C string  con el valor a escribir.
		\param Int con el largo del string value.
		\return True si se pudo realizar la escritura, false en caso contrario.
	*/
	bool writeValue (char* label, int lengthLabel, char* value, int lengthValue);

	/// Funci�n que escribe una fila con valores num�ricos.
	/** Funci�n que escribe una fila con valores num�ricos separados por coma, para
		esto recibe un puntero al vector con los n�meros, y la cantidad de n�meros
		que contiene dicho vector.
		\param Vector con los n�meros a escribir.
		\param Cantidad de n�meros que contiene el vector.
		\return True si se pudo realizar la escritura, false en caso contrario.
	*/
	bool writeRowOfNumbers (float* numbers, int numNumbers);
private:
	
	/// Nombre completo del archivo, que incluye el path y la extensi�n.
	string fullFileName;

	/// Caracter de fin de l�nea que se utiliza: \n.
	/* Caracter de fin de l�nea que se utiliza: \n. Se podr�a agregar para que sea seleccionable para el usuario
	o para que lo asigne dependiendo el sistema operativo. */
	string eol;

	/// Objeto ofstream para la escritura en el archivo de log.
	ofstream fileStream;

};

#endif