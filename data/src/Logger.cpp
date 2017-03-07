/**
	\file Logger.cpp
	\brief Archivo con los métodos de la clase Logger.
	
	\author Martín Belzunce (martin.sure@gmail.com)
	\date 2010.12.06
	\version 1.0.0
	
*/

#include <Logger.h>

Logger::Logger(string fullFileName)
{
	/// Abro y creo el archivo.
	fileStream.open(fullFileName.c_str(), ios::out);
	/// Asigno el caracter de fin de linea.
	eol.assign("\r\n");
}

Logger::~Logger()
{
	/// Si está abierto lo cierro.
	if(fileStream.is_open())
	{
		fileStream.close();
	}
}

bool Logger::writeLine(string line)
{
	/// Si no está abierto lo abro en modo append antes de escribir.
	if(!fileStream.is_open())
	{
		fileStream.open(this->fullFileName.c_str(), ios::app);
	}

	/// Le agrego el caracter de fin de linea:
	line.append(this->eol);
	fileStream.write(line.c_str(), line.length());

	/// Para que se envíe directamente al archivo y no quede esperando en un buffer a que se llene.
	fileStream.flush();
	return true;
}

bool Logger::writeLine(char* line, int length)
{
	/// Si no está abierto lo abro en modo append antes de escribir.
	if(!fileStream.is_open())
	{
		fileStream.open(this->fullFileName.c_str(), ios::app);
	}

	/// Le agrego el caracter de fin de linea:
	strcat(line, this->eol.c_str());
	fileStream.write(line, strlen(line));

	/// Para que se envíe directamente al archivo y no quede esperando en un buffer a que se llene.
	fileStream.flush();
	return true;
}

bool Logger::writeValue (string label, string value)
{
	/// Si no está abierto lo abro en modo append antes de escribir.
	if(!fileStream.is_open())
	{
		fileStream.open(this->fullFileName.c_str(), ios::app);
		/// Si hubo un error devuelvo false.
		if(fileStream.bad())
			return false;
	}
	label.append(": ");
	label.append(value);
	/// Le agrego el caracter de fin de linea:
	label.append(this->eol);
	fileStream.write(label.c_str(), label.length());
	/// Si hubo un error devuelvo false.
	if(fileStream.bad())
		return false;

	/// Para que se enviíe directamente al archivo y no quede esperando en un buffer a que se llene.
	fileStream.flush();
	return true;
}

bool Logger::writeValue (char* label, int lengthLabel, char* value, int lengthValue)
{
  char line[512];
  if(lengthLabel+lengthValue+2)
  {
    /// Me pasé de la cantidad de líneas por renglón.
    return false;
  }
	/// Si no está abierto lo abro en modo append antes de escribir.
	if(!fileStream.is_open())
	{
		fileStream.open(this->fullFileName.c_str(), ios::app);
		/// Si hubo un error devuelvo false.
		if(fileStream.bad())
			return false;
	}
	strcpy(line, label);
	strcat(line, ": ");
	strcat(line, value);
	/// Le agrego el caracter de fin de linea:
	strcat(line, this->eol.c_str());
	fileStream.write(line, strlen(line));
	/// Si hubo un error devuelvo false.
	if(fileStream.bad())
		return false;

	/// Para que se envíe directamente al archivo y no quede esperando en un buffer a que se llene.
	fileStream.flush();
	return true;
}

bool Logger::writeRowOfNumbers (float* numbers, int numNumbers)
{
	string line;
	/// Genero un string con los números:
	ostringstream stringValues;
	/// Inicializo el primer valor
	stringValues << numbers[0];
	for(int i = 1; i < numNumbers; i++)
	{
		stringValues << ", " << numbers[i];
	}
	line.assign(stringValues.str());
	line.append(this->eol);
	/// Si no está abierto lo abro en modo append antes de escribir.
	if(!fileStream.is_open())
	{
		fileStream.open(this->fullFileName.c_str(), ios::app);
		/// Si hubo un error devuelvo false.
		if(fileStream.bad())
			return false;
	}

	fileStream.write(line.c_str(),line.length());
	/// Si hubo un error devuelvo false.
	if(fileStream.bad())
		return false;

	/// Para que se envíe directamente al archivo y no quede esperando en un buffer a que se llene.
	fileStream.flush();
	return true;
}