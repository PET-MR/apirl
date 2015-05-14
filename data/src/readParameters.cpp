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


#include <readParameters.h>


using namespace std;
using	std::string;




int getSensitivityFromFile (string mlemFilename, string cmd, bool* bSensitivityFromFile, string* sensitivityFilename)
{
  int errorCode;
  char returnValue[256];	// string en el que se recibe el valor de un keyword en la lectura del archivo de parámetros.
  char errorMessage[300];	// string de error para la función de lectura de archivo de parámetros.
  *bSensitivityFromFile = false;
  if((errorCode=parametersFile_read((char*)mlemFilename.c_str(), (char*)cmd.c_str(), "sensitivity filename", returnValue, errorMessage)) != 0)
  {
	  // Hubo un error. Salgo del comando.
	  // Si no encontró el keyoword, está bien porque era opcional, cualquier otro código de error
	  // signfica que hubo un error.
	  if(errorCode == PMF_KEY_NOT_FOUND)
	  {
	    // No está la keyword, como era opcional se carga con su valor por default.
	    *sensitivityFilename = "";
	    *bSensitivityFromFile = false;
	  }
	  else
	  {
	    cout<<"Error "<<errorCode<<" en el archivo de parámetros. Mirar la documentación de los códigos de errores."<<endl;
	    return -1;
	  }
  }
  else
  {
    *bSensitivityFromFile = true;
    (*sensitivityFilename).assign(returnValue);
  }
  return 0;
}

/* Función que lee las opciones con el intervalo en que se guardan las imagenes de cada iteración, y también si se deben
 * guardar las proyecciones intermedias. Devuelve 0 si pudo leer la configuración correctamente si no -1.
 * Falta poner los comentarios bien con doxygen.
 * 
 */
int getSaveIntermidiateIntervals (string mlemFilename, string cmd, int* saveIterationInterval, bool* saveIntermediateData)
{
  int errorCode;
  char returnValue[256];	// string en el que se recibe el valor de un keyword en la lectura del archivo de parámetros.
  char errorMessage[300];	// string de error para la función de lectura de archivo de parámetros.
  *saveIterationInterval = 0;
  *saveIntermediateData = 0;
// "save estimates at iteration intervals"
  if((errorCode=parametersFile_read((char*)mlemFilename.c_str(), (char*)cmd.c_str(), "save estimates at iteration intervals", returnValue, errorMessage)) != 0)
  {
	  // Hubo un error. Salgo del comando.
	  // Si no encontró el keyoword, está bien porque era opcional, cualquier otro código de error
	  // signfica que hubo un error.
	  if(errorCode == PMF_KEY_NOT_FOUND)
	  {
	    // No está la keyword, como era opcional se carga con su valor por default.
	    *saveIterationInterval = 0;
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
    *saveIterationInterval = atoi(returnValue);
    // Si setearon el save iteration interval, también tengo que ver si está el "save estimated projections and backprojected image":
    if((errorCode=parametersFile_read((char*)mlemFilename.c_str(), (char*)cmd.c_str(), "save estimated projections and backprojected image", returnValue, errorMessage)) != 0)
    {
	  // Hubo un error. Salgo del comando.
	  // Si no encontró el keyoword, está bien porque era opcional, cualquier otro código de error
	  // signfica que hubo un error.
	  if(errorCode == PMF_KEY_NOT_FOUND)
	  {
	    // No está la keyword, como era opcional se carga con su valor por default.
	    *saveIntermediateData = 0;
	  }
	  else
	  {
	    cout<<"Error "<<errorCode<<" en el archivo de parámetros. Mirar la documentación de los códigos de errores."<<endl;
	    return -1;
	  }
    }
    else
    {
	  *saveIntermediateData = (bool)atoi(returnValue);
    }
    return 0;
  }
}

/* Función que obtiene los nombres (solo los nombres!) del projecctor y backprojector
 * 
 */
int getProjectorBackprojectorNames(string mlemFilename, string cmd, string* strForwardprojector, string* strBackprojector)
{
  int errorCode;
  char returnValue[256];	// string en el que se recibe el valor de un keyword en la lectura del archivo de parámetros.
  char errorMessage[300];	// string de error para la función de lectura de archivo de parámetros.
  // Lectura de Proyectores, tanto para forwardprojection como para backprojection.
  if((errorCode=parametersFile_read((char*)mlemFilename.c_str(), (char*)cmd.c_str(), "forwardprojector", returnValue, errorMessage)) != 0)
  {
    if(errorCode == PMF_KEY_NOT_FOUND)
    {
      cout<<"Error "<<errorCode<<" en el archivo de parámetros. No se encuentra la definición del forwardprojector."<<endl;
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
    (*strForwardprojector).assign(returnValue);
  }
  if((errorCode=parametersFile_read((char*)mlemFilename.c_str(), (char*)cmd.c_str(), "backprojector", returnValue, errorMessage)) != 0)
  {
    if(errorCode == PMF_KEY_NOT_FOUND)
    {
      cout<<"Error "<<errorCode<<" en el archivo de parámetros. No se encuentra la definición del backprojector."<<endl;
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
    (*strBackprojector).assign(returnValue);
  }
  return 0;
}

/* Función que obtiene los parámetros de los proyectores de siddon.
 */
int getSiddonProjectorParameters(string mlemFilename, string cmd, int* numSamples)
{
  int errorCode;
  char returnValue[256];	// string en el que se recibe el valor de un keyword en la lectura del archivo de parámetros.
  char errorMessage[300];	// string de error para la función de lectura de archivo de parámetros.
  
  // Debo leer el parámetro que tiene: "siddon number of samples on the detector".
  if((errorCode=parametersFile_read((char*)mlemFilename.c_str(), (char*)cmd.c_str(), "siddon number of samples on the detector", returnValue, errorMessage)) != 0)
  {
    // Hubo un error. Salgo del comando.
    if(errorCode == PMF_KEY_NOT_FOUND)
    {
      cout<<"No se encontró el parámetro ""siddon number of samples on the detector"", se toma el valor por defecto."<<endl;
      return -1;
    }
    else
    {
      cout<<"Error "<<errorCode<<" en el archivo de parámetros. Mirar la documentación de los códigos de errores."<<endl;
      return -1;
    }
  }
  *numSamples = atoi(returnValue);
  return 0;
}

/* Función que obtiene los parámetros de los proyectores basados en rotación. Por ahora
 * el único parámetro a cargar es el método de interpolación.
 */
int getRotationBasedProjectorParameters(string mlemFilename, string cmd, string *interpMethod)
{
  int errorCode;
  char returnValue[256];	// string en el que se recibe el valor de un keyword en la lectura del archivo de parámetros.
  char errorMessage[300];	// string de error para la función de lectura de archivo de parámetros.
  
  // Debo leer el parámetro que tiene: "interpolation_method".
  if((errorCode=parametersFile_read((char*)mlemFilename.c_str(), (char*)cmd.c_str(), "interpolation_method", returnValue, errorMessage)) != 0)
  {
    // Hubo un error. Salgo del comando.
    if(errorCode == PMF_KEY_NOT_FOUND)
    {
      cout<<"No se encontró el parámetro ""interpolation_method"", el cual es obligatorio "
      "para el proyector RotationBasedProjector."<<endl;
      return -1;
    }
    else
    {
      cout<<"Error "<<errorCode<<" en el archivo de parámetros. Mirar la documentación de los códigos de errores."<<endl;
      return -1;
    }
  }

  if((!strcmp(returnValue, "nearest"))||(!strcmp(returnValue, "bilinear"))||(!strcmp(returnValue, "bicubic")))
  {
	  (*interpMethod).assign(returnValue);
  }
  else
  {
    cout<<"Error "<<errorCode<<" en el archivo de parámetros. Método de interpolación inexistente para RotationBasedProjector."<<endl;
    return -1;
  }
  return 0;
}

/*
 * 
 * Condición que por ahora no se usa
 */
int getPositivityCondition (string mlemFilename, string cmd)
{
  int errorCode;
  char returnValue[256];	// string en el que se recibe el valor de un keyword en la lectura del archivo de parámetros.
  char errorMessage[300];	// string de error para la función de lectura de archivo de parámetros.
  // "enforce initial positivity condition"
  if((errorCode=parametersFile_read((char*)mlemFilename.c_str(), (char*)cmd.c_str(), "enforce initial positivity condition", returnValue, errorMessage)) != 0)
  {
	  // Hubo un error. Salgo del comando.
	  // Si no encontró el keyoword, está bien porque era opcional, cualquier otro código de error
	  // signfica que hubo un error.
	  if(errorCode == PMF_KEY_NOT_FOUND)
	  {
	    // No está la keyword, como era opcional se carga con su valor por default.
	  }
	  else
	  {
	    cout<<"Error "<<errorCode<<" en el archivo de parámetros. Mirar la documentación de los códigos de errores."<<endl;
	    return -1;
	  }
  }
  else
  {
    
  }
}
	
	
/*
 * 
 */
int getCylindricalScannerParameters(string mlemFilename, string cmd, float* radiusFov_mm, float* zFov_mm, float* radiusScanner_mm)
{
  int errorCode;
  char returnValue[256];	// string en el que se recibe el valor de un keyword en la lectura del archivo de parámetros.
  char errorMessage[300];	// string de error para la función de lectura de archivo de parámetros.
  // Radio del scanner:
  if((errorCode=parametersFile_read((char*)mlemFilename.c_str(), (char*)cmd.c_str(), "cylindrical pet radius (in mm)", returnValue, errorMessage)) != 0)
  {
	  // Hubo un error. Salgo del comando.
	  // Si no encontró el keyoword, está bien porque era opcional, cualquier otro código de error
	  // signfica que hubo un error.
	  if(errorCode == PMF_KEY_NOT_FOUND)
	  {
	    // No está la keyword, como era opcional se carga con su valor por default.
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
    *radiusScanner_mm = atof(returnValue);
  }
  
  // Radio del Fov:
  if((errorCode=parametersFile_read((char*)mlemFilename.c_str(), (char*)cmd.c_str(), "radius fov (in mm)", returnValue, errorMessage)) != 0)
  {
	  // Hubo un error. Salgo del comando.
	  // Si no encontró el keyoword, está bien porque era opcional, cualquier otro código de error
	  // signfica que hubo un error.
	  if(errorCode == PMF_KEY_NOT_FOUND)
	  {
	    // No está la keyword, como era opcional se carga con su valor por default.
	  
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
    *radiusFov_mm = atof(returnValue);
  }
  
  // largo del scanner:
  if((errorCode=parametersFile_read((char*)mlemFilename.c_str(), (char*)cmd.c_str(), "axial fov (in mm)", returnValue, errorMessage)) != 0)
  {
	  // Hubo un error. Salgo del comando.
	  // Si no encontró el keyoword, está bien porque era opcional, cualquier otro código de error
	  // signfica que hubo un error.
	  if(errorCode == PMF_KEY_NOT_FOUND)
	  {
	    // No está la keyword, como era opcional se carga con su valor por default.
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
    *zFov_mm = atof(returnValue);
  }
  
  return 0;
}

int getNumberOfSubsets(string mlemFilename, string cmd, float* numberOfSubsets)
{
  int errorCode;
  char returnValue[256];	// string en el que se recibe el valor de un keyword en la lectura del archivo de parámetros.
  char errorMessage[300];	// string de error para la función de lectura de archivo de parámetros.
  
  if(cmd.compare("OSEM"))
  {
    // El subsets es solo válido para OSEM:
    return -1;
  }
  // Radio del scanner:
  if((errorCode=parametersFile_read((char*)mlemFilename.c_str(), (char*)cmd.c_str(), "number of subsets", returnValue, errorMessage)) != 0)
  {
    // Hubo un error. Salgo del comando.
    // Si no encontró el keyoword, está bien porque era opcional, cualquier otro código de error
    // signfica que hubo un error.
    if(errorCode == PMF_KEY_NOT_FOUND)
    {
      // No está la keyword, como era opcional se carga con su valor por default.
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
    *numberOfSubsets = atof(returnValue);
  }
  return 0;
}

int getArPetParameters(string mlemFilename, string cmd, float* radiusFov_mm, float* zFov_mm, float* blindArea_mm, int* minDiffDetectors)
{
  int errorCode;
  char returnValue[256];	// string en el que se recibe el valor de un keyword en la lectura del archivo de parámetros.
  char errorMessage[300];	// string de error para la función de lectura de archivo de parámetros.
  
  // Radio del Fov:
  if((errorCode=parametersFile_read((char*)mlemFilename.c_str(), (char*)cmd.c_str(), "radius fov (in mm)", returnValue, errorMessage)) != 0)
  {
    // Hubo un error. Salgo del comando.
    // Si no encontró el keyoword, está bien porque era opcional, cualquier otro código de error
    // signfica que hubo un error.
    if(errorCode == PMF_KEY_NOT_FOUND)
    {
      // No está la keyword, como era opcional se carga con su valor por default.
    
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
    *radiusFov_mm = atof(returnValue);
  }
  
  
  // Radio del scanner:
  if((errorCode=parametersFile_read((char*)mlemFilename.c_str(), (char*)cmd.c_str(), "ArPet blind area (in mm)", returnValue, errorMessage)) != 0)
  {
	  // Hubo un error. Salgo del comando.
	  // Si no encontró el keyoword, está bien porque era opcional, cualquier otro código de error
	  // signfica que hubo un error.
	  if(errorCode == PMF_KEY_NOT_FOUND)
	  {
	    // No está la keyword, como era opcional se carga con su valor por default.
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
    *blindArea_mm = atof(returnValue);
  }
  
  // Radio del scanner:
  if((errorCode=parametersFile_read((char*)mlemFilename.c_str(), (char*)cmd.c_str(), "ArPet minimum difference between detectors", returnValue, errorMessage)) != 0)
  {
	  // Hubo un error. Salgo del comando.
	  // Si no encontró el keyoword, está bien porque era opcional, cualquier otro código de error
	  // signfica que hubo un error.
	  if(errorCode == PMF_KEY_NOT_FOUND)
	  {
	    // No está la keyword, como era opcional se carga con su valor por default.
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
    *minDiffDetectors = atoi(returnValue);
  }
  
  
  // largo del scanner:
  if((errorCode=parametersFile_read((char*)mlemFilename.c_str(), (char*)cmd.c_str(), "axial fov (in mm)", returnValue, errorMessage)) != 0)
  {
	  // Hubo un error. Salgo del comando.
	  // Si no encontró el keyoword, está bien porque era opcional, cualquier otro código de error
	  // signfica que hubo un error.
	  if(errorCode == PMF_KEY_NOT_FOUND)
	  {
	    // No está la keyword, como era opcional se carga con su valor por default.
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
    *zFov_mm = atof(returnValue);
  }
  
  return 0;
}

int getCorrectionSinogramNames(string mlemFilename, string cmd, string* acfFilename, string* estimatedRandomsFilename, string* estimatedScatterFilename)
{
  int errorCode;
  char returnValue[256];	// string en el que se recibe el valor de un keyword en la lectura del archivo de parámetros.
  char errorMessage[300];	// string de error para la función de lectura de archivo de parámetros.
  // Lectura de Proyectores, tanto para forwardprojection como para backprojection.
  if((errorCode=parametersFile_read((char*)mlemFilename.c_str(), (char*)cmd.c_str(), "attenuation correction factors", returnValue, errorMessage)) != 0)
  {
    if(errorCode == PMF_KEY_NOT_FOUND)
    {
      // Si no lo encuentra pongo el string vacío, de esta forma indico que no se realiza la corrección por atenuación:
      *acfFilename = "";
    }
    else
    {
      cout<<"Error "<<errorCode<<" en el archivo de parámetros. Mirar la documentación de los códigos de errores."<<endl;
      return -1;
    }
  }
  else
  {
    (*acfFilename).assign(returnValue);
  }
  if((errorCode=parametersFile_read((char*)mlemFilename.c_str(), (char*)cmd.c_str(), "randoms correction sinogram", returnValue, errorMessage)) != 0)
  {
    if(errorCode == PMF_KEY_NOT_FOUND)
    {
      // Si no lo encuentra pongo el string vacío, de esta forma indico que no se realiza la corrección por randoms:
      *estimatedRandomsFilename = "";
    }
    else
    {
      cout<<"Error "<<errorCode<<" en el archivo de parámetros. Mirar la documentación de los códigos de errores."<<endl;
      return -1;
    }
  }
  else
  {
    (*estimatedRandomsFilename).assign(returnValue);
  }
  
  if((errorCode=parametersFile_read((char*)mlemFilename.c_str(), (char*)cmd.c_str(), "scatter correction sinogram", returnValue, errorMessage)) != 0)
  {
    if(errorCode == PMF_KEY_NOT_FOUND)
    {
      // Si no lo encuentra pongo el string vacío, de esta forma indico que no se realiza la corrección de scater:
      *estimatedScatterFilename = "";
    }
    else
    {
      cout<<"Error "<<errorCode<<" en el archivo de parámetros. Mirar la documentación de los códigos de errores."<<endl;
      return -1;
    }
  }
  else
  {
    (*estimatedScatterFilename).assign(returnValue);
  }
  return 0;
}


int getNormalizationSinogramName(string mlemFilename, string cmd, string* normFilename)
{
  int errorCode;
  char returnValue[256];	// string en el que se recibe el valor de un keyword en la lectura del archivo de parámetros.
  char errorMessage[300];	// string de error para la función de lectura de archivo de parámetros.
  // Lectura de Proyectores, tanto para forwardprojection como para backprojection.
  if((errorCode=parametersFile_read((char*)mlemFilename.c_str(), (char*)cmd.c_str(), "normalization correction factors", returnValue, errorMessage)) != 0)
  {
    if(errorCode == PMF_KEY_NOT_FOUND)
    {
      // Si no lo encuentra pongo el string vacío, de esta forma indico que no se realiza la corrección por atenuación:
      *normFilename = "";
    }
    else
    {
      cout<<"Error "<<errorCode<<" en el archivo de parámetros. Mirar la documentación de los códigos de errores."<<endl;
      return -1;
    }
  }
  else
  {
    (*normFilename).assign(returnValue);
  }
  
  return 0;
}