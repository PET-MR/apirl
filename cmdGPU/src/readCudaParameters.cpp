/**
	\file readParameters.cpp
	\brief Archivo con funciones para leer los parámetros de los ejecutables.

	Este archivo tiene funciones para leer los parámetros que necesitan los ejecutables, como los proyectores,
	backprojectores, etc.
	
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2010.08.30
	\version 1.1.0
*/


#include <readCudaParameters.h>


using namespace std;
using	std::string;


bool ProcessBlockSizeString(char* strBlockSize, dim3* blockSize)
{
  char* aux = strrchr( strBlockSize, '{'); // Este es el inicio del buffer
  char* end = strrchr( strBlockSize, '}'); // Este es el inicio del buffer
  if ((aux == NULL)||(end == NULL)){
    cout << "No se encontraron los {} de la lista de sinogramas por segmento." << endl; return false;}
  // Si llegué hasta acá pongo el fin del string donde estaba la llave de cierre y saltea una posición donde estaba la de apertura:
  aux++;
  (*end) = '\0';
  // Ahora obtengo todos los strings separados por coma, para eso uso la función strtok:
  char* auxElemento = strtok (aux," ,");
  blockSize->x = atoi(auxElemento);
  if(auxElemento == NULL){
      cout << "Faltan elmentos en el block size." << endl; return false;}
  // El y:
  auxElemento = strtok (aux," ,");
  blockSize->y = atoi(auxElemento);
  if(auxElemento == NULL){
      cout << "Faltan elmentos en el block size." << endl; return false;}
  // El z:
  auxElemento = strtok (aux," ,");
  blockSize->z = atoi(auxElemento);
  if(auxElemento == NULL){
      cout << "Faltan elmentos en el block size." << endl; return false;}

  return true;
}

int getProjectorBlockSize (string mlemFilename, string cmd, dim3* projectorBlockSize)
{
  int errorCode;
  char returnValue[256];	// string en el que se recibe el valor de un keyword en la lectura del archivo de parámetros.
  char errorMessage[300];	// string de error para la función de lectura de archivo de parámetros.
  if((errorCode=parametersFile_read((char*)mlemFilename.c_str(), (char*)cmd.c_str(), "projector block size", returnValue, errorMessage)) != 0)
  {
    // Hubo un error. Salgo del comando.
    // Si no encontró el keyoword, está bien porque era opcional, cualquier otro código de error
    // signfica que hubo un error.
    if(errorCode == PMF_KEY_NOT_FOUND)
    {
      // No está la keyword, como era opcional se carga con su valor por default.
      cout<<"Error "<<errorCode<<" en el archivo de parámetros. Mirar la documentación de los códigos de errores."<<endl;
      return -1;
    }
    else
    {
      cout<<"Error "<<errorCode<<" en el archivo de parámetros. Mirar la documentación de los códigos de errores."<<endl;
      return -1;
    }
  }
  else
  {
    if(!ProcessBlockSizeString(returnValue, projectorBlockSize))
    {
      return -1;
    }
  }
  return 0;
}

int getPixelUpdateBlockSize (string mlemFilename, string cmd, dim3* updateBlockSize)
{
  int errorCode;
  char returnValue[256];	// string en el que se recibe el valor de un keyword en la lectura del archivo de parámetros.
  char errorMessage[300];	// string de error para la función de lectura de archivo de parámetros.
  if((errorCode=parametersFile_read((char*)mlemFilename.c_str(), (char*)cmd.c_str(), "projector block size", returnValue, errorMessage)) != 0)
  {
    // Hubo un error. Salgo del comando.
    // Si no encontró el keyoword, está bien porque era opcional, cualquier otro código de error
    // signfica que hubo un error.
    if(errorCode == PMF_KEY_NOT_FOUND)
    {
      // No está la keyword, como era opcional se carga con su valor por default.
      cout<<"Error "<<errorCode<<" en el archivo de parámetros. Mirar la documentación de los códigos de errores."<<endl;
      return -1;
    }
    else
    {
      cout<<"Error "<<errorCode<<" en el archivo de parámetros. Mirar la documentación de los códigos de errores."<<endl;
      return -1;
    }
  }
  else
  {
    if(!ProcessBlockSizeString(returnValue, updateBlockSize))
    {
      return -1;
    }
  }
  return 0;
}

int getBackprojectorBlockSize (string mlemFilename, string cmd, dim3* backprojectorBlockSize)
{
  int errorCode;
  char returnValue[256];	// string en el que se recibe el valor de un keyword en la lectura del archivo de parámetros.
  char errorMessage[300];	// string de error para la función de lectura de archivo de parámetros.
  if((errorCode=parametersFile_read((char*)mlemFilename.c_str(), (char*)cmd.c_str(), "projector block size", returnValue, errorMessage)) != 0)
  {
    // Hubo un error. Salgo del comando.
    // Si no encontró el keyoword, está bien porque era opcional, cualquier otro código de error
    // signfica que hubo un error.
    if(errorCode == PMF_KEY_NOT_FOUND)
    {
      // No está la keyword, como era opcional se carga con su valor por default.
      cout<<"Error "<<errorCode<<" en el archivo de parámetros. Mirar la documentación de los códigos de errores."<<endl;
      return -1;
    }
    else
    {
      cout<<"Error "<<errorCode<<" en el archivo de parámetros. Mirar la documentación de los códigos de errores."<<endl;
      return -1;
    }
  }
  else
  {
    if(!ProcessBlockSizeString(returnValue, backprojectorBlockSize))
    {
      return -1;
    }
  }
  return 0;
}


int getGpuId(string mlemFilename, string cmd, int* gpuId)
{
  int errorCode;
  char returnValue[256];	// string en el que se recibe el valor de un keyword en la lectura del archivo de parámetros.
  char errorMessage[300];	// string de error para la función de lectura de archivo de parámetros.
// "save estimates at iteration intervals"
  if((errorCode=parametersFile_read((char*)mlemFilename.c_str(), (char*)cmd.c_str(), "gpu id", returnValue, errorMessage)) != 0)
  {
    // Hubo un error. Salgo del comando.
    // Si no encontró el keyoword, está bien porque era opcional, cualquier otro código de error
    // signfica que hubo un error.
    if(errorCode == PMF_KEY_NOT_FOUND)
    {
      // No está la keyword, como era opcional se carga con su valor por default.
      *gpuId = 0;
      return 0;
    }
    else
    {
      cout<<"Error "<<errorCode<<" en el archivo de parámetros. Mirar la documentación de los códigos de errores."<<endl;
      return -1;
    }
  }
  else
  {
    *gpuId = atoi(returnValue);
    return 0;
  }
}

