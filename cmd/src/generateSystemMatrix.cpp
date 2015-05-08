/**
	\file generateSystemMatrix.cpp
	\brief Ejecutable para generar una system matrix para un proyector dado.

	Este archivo de código genera el ejecutable de un comando que calcula la system matrix
	para un proyector, imagen y proyecciones dadas. Al momento solo se utiliza para Sinogram2Dtgs
	porque para otros sinogramas tendría un tamaño impracticable.
	
	\par Ejemplo de archivo de configuración de generación de System Matrix genSMR.par:
	\code
	    generateSystemMatrix Parameters :=
	    ; Ejemplo de archivo de configuración de reconstrucción MLEM.
	    projection type := Sinogram2Dtgs
	    Sinogram2Dtgs Parameters :=		
		  diameter_of_fov (in mm) := 600
		  distance_cristal_to_center_of_fov (in mm) := 400
		  length_of_colimator (in mm) := 100
		  diameter_of_colimator (in mm) := 20
		End Sinogram2Dtgs Parameters :=
	    projection file := test.hs
	    image file := some_image
		; Projectors:
		projector := ConeOfResponse
	    output filename prefix := test_SMR
	    END :=
	\endcode
	\todo Implementarlo para otros tipos de proyecciones además de Sinogram2Dtgs. Y permitir que la SMR sea del tipo sparse. 
		  Para esto habría que agregar alguna librería del tipo sparse.
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2011.11.09
	\version 1.1.0
*/

#include <stdio.h>
#include <iostream>
#include <string.h>
#include <Michelogram.h>
#include <ParametersFile.h>
#include <Projector.h>
#include <SiddonProjector.h>
#include <ConeOfResponseProjector.h>
#include <Images.h>
#include <Geometry.h>
#include <string>

#define FOV_axial 162
#define FOV_radial 582
#define FIXED_KEYS 5
using namespace std;
using	std::string;
/**
	\fn void main (int argc, char *argv[])
	\brief Ejecutable que genera una SMR, para eso recibe como parámetro el nombre del archivo de configuración de reconstrucción.
	
	Este comando genera una System Matrix. Para esto se le debe indicar un tamaño de imagen, un tamaño y tipo de sinograma, y por
	último se debe seleccionar un projector. Este último es el modelo que se utilizará para generar el valor de cada elemento de la
	SMR. Por ahora los proyectores disponibles son:
	  -Siddon.
	  -ConeOfResponse.
	El archivo de parámetros tiene un formato similar a los archivos interfile, se diferencia en que la primera línea debe ser
	"generateSystemMatrix Parameter :=" y debe finalizar con un "END :=". Este formato está basado en el propuesto por STIR para configurar sus
	métodos de reconstrucción.
	Cada parámetro es ingresado en el archivo a través de keyword := value, siendo keyword el nombre del parámetros y value el valor
	que se le asigna. Hay campos obligatorios, y otros opcionales.
	Campos Obligatorios:
		- "projection type" : tipo de proyección del sistema. Los valores posibles son: Sinogram2Dtgs. (Hay que agregar el resto)
		- "projection file" : nombre del archivo header (*.hs) de una proyección del tipo establecido en formato interfile. Solo
							  se utiliza para obtener el tamaño del singograma.
		- "image file" : nombre del archivo header (*.hs) de la imagen de salida del sistema en formato interfile. Se utiliza para obtener
						 el tamaño de la imagen de salida del sistema.
		- "output filename prefix" : prefijo para los nombres de archivo de salida (principalmente nombres de las imágenes).
		- "projector" :  proyector utilizado para obtener la matriz del sistema.
	Luego hay parámetros que son específicos a cada tipo de dato. Para el input type := Sinogram2Dtgs se deben cargar los siguientes parámetros:
		Sinogram2Dtgs Parameters :=		
		  diameter_of_fov (in mm) := 600
		  distance_cristal_to_center_of_fov (in mm) := 400
		  length_of_colimator (in mm) := 100
		  diameter_of_colimator (in mm) := 20
		End Sinogram2Dtgs Parameters :=
	  Siendo:
		- "diameter_of_fov (in mm)" : diámetro del field of view.
		- "distance_cristal_to_center_of_fov (in mm)" : la distnacia en mm de la superficie del cristal al centro del fov.
		- "length_of_colimator (in mm)" : largo del colimador en mm.
		- "diameter_of_colimator (in mm)" : diámetro del agujero del colimador en mm.
	\par Ejemplo de Archivo de parámetro .par
	\code
	  generateSystemMatrix Parameters :=
	    ; Ejemplo de archivo de configuración de reconstrucción MLEM.
	    projection type := Sinogram2Dtgs
	    Sinogram2Dtgs Parameters :=		
		  diameter_of_fov (in mm) := 600
		  distance_cristal_to_center_of_fov (in mm) := 400
		  length_of_colimator (in mm) := 100
		  diameter_of_colimator (in mm) := 20
		End Sinogram2Dtgs Parameters :=
	    projection file := test.hs
	    image file := some_image
		; Projectors:
		projector := ConeOfResponse
	    output filename prefix := test_SMR
	  END :=
	\endcode

	@param argc Cantidad de argumentos de entrada
	@param argv Puntero a vector con los argumentos de entrada. El comando debe ejecutarse con el nombre del archivo de parámetros como argumento.
	@return 0 si no uhbo errores, 1  si falló la operación.
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2011.11.09
	\version 1.0.0
*/	

int main (int argc, char *argv[])
{
	char errorMessage[300];	// string de error para la función de lectura de archivo de parámetros.
	char returnValue[256];	// string en el que se recibe el valor de un keyword en la lectura del archivo de parámetros.
	char** keyWords;  // múltiples keywords para la función de lectura de múltiples keys.
	char** multipleReturnValue; // array de strings para la función de multples keys.
	int	errorCode;
	int nx, ny, nz, n_voxels, resolution, numIterations, file, Recon;
	unsigned int hTimer;
	const char* pathSensImage;
	double timerValue;
	SizeImage MySizeVolume;
	char NombreRecon[50];
	string parameterFileName;	// string para el Nombre de Archivo de parámetros.
	string projectionType;
	string projectionFilename;	// string para el Nombre del archivo de header del sinograma.
	string outputPrefix;	// prefijo para los nombres de los archivos de salida.
	string imageFilename;	// nombre del archivo de la imagen inicial.
	string strProjector;
	Projector* projector;
	Image* image;
	
	// Asigno la memoria para los punteros dobles, para el array de strings.
	keyWords = (char**)malloc(sizeof(*keyWords)*FIXED_KEYS);
	multipleReturnValue = (char**)malloc(sizeof(*multipleReturnValue)*FIXED_KEYS);
	for(int j = 0; j < FIXED_KEYS; j++)
	{
	  keyWords[j] = (char*) malloc(sizeof(char)*MAX_KEY_LENGTH);
	  multipleReturnValue[j] = (char*) malloc(sizeof(char)*MAX_KEY_LENGTH);
	}
	
	// Verificación de que se llamo al comando con el nombre de archivo de parámetros como argumento.
	if(argc != 2)
	{
		cout << "El comando generateSystemMatrix debe llamarse indicando el archivo de Parámetros del sistema: genSMR.par." << endl;
		return -1;
	}
	// Los parámetros de reconstrucción son los correctos.
	// Se verifica que el archivo tenga la extensión .par.
	parameterFileName.assign(argv[1]);
	//strcpy(parameterFileName, argv[1]);
	if(parameterFileName.compare(parameterFileName.length()-4, 4, ".par"))
	{
		// El archivo de parámetro no tiene la extensión .par.
		cout<<"El archivo de parámetros no tiene la extensión .par."<<endl;
		return -1;
	}

	// Leo cada uno de los campos del archivo de parámetros. Para esto utilizo la función parametersFile_readMultipleKeys
	// que  me permite leer múltiples keys en un único llamado a función. Para esto busco los keywords que forman 
	// parte de los campos obligatorios, los opcionales los hago de a uno por vez.
	strcpy(keyWords[0], "projection type"); 
	strcpy(keyWords[1], "projection file"); 
	strcpy(keyWords[2], "image file"); 
	strcpy(keyWords[3], "output filename prefix"); 
	strcpy(keyWords[4], "projector"); 
	if((errorCode=parametersFile_readMultipleKeys((char*)parameterFileName.c_str(), (char*)"generateSystemMatrix", (char**)keyWords, FIXED_KEYS, (char**)multipleReturnValue, errorMessage)) != 0)
	{
		// Hubo un error. Salgo del comando.
		cout<<"Error "<<errorCode<<" en el archivo de parámetros. Mirar la documentación de los códigos de errores."<<endl;
		return -1;
	}
	projectionType.assign(multipleReturnValue[0]);
	projectionFilename.assign(multipleReturnValue[1]);
	imageFilename.assign(multipleReturnValue[2]);
	outputPrefix.assign(multipleReturnValue[3]);
	strProjector.assign(multipleReturnValue[4]);
	//outputFileName.assign(returnValue);

	// Inicializo el proyector a utilizar:
	if(strProjector.compare("Siddon") == 0)
	{
	  projector = (Projector*)new SiddonProjector();
	}
	else if(strProjector.compare("ConeOfResponse") == 0)
	{
	  // Debo leer el parámetro que tiene: "number_of_points_on_detector".
	  if((errorCode=parametersFile_read((char*)parameterFileName.c_str(), (char*)"generateSystemMatrix", (char*)"number_of_points_on_detector", (char*)returnValue, (char*)errorMessage)) != 0)
	  {
		  // Hubo un error. Salgo del comando.
		  if(errorCode == PMF_KEY_NOT_FOUND)
		  {
			cout<<"No se encontró el parámetro ""number_of_points_on_detector"", el cual es obligatorio "
			"para el proyector ConeOfResponse."<<endl;
			return -1;
		  }
		  else
		  {
			cout<<"Error "<<errorCode<<" en el archivo de parámetros. Mirar la documentación de los códigos de errores."<<endl;
			return -1;
		  }
	  }
	  int numPointsOnDetector = atoi(returnValue);
	  
	  projector = (Projector*)new ConeOfResponseProjector(numPointsOnDetector);
	}
	else
	{
	  cout<<"Projector inválido."<<endl;
	  return -1;
	}	
	
	// Lectura de los parámetros opcionales de reconstrucción MLEM:
	// "Parametros de Sinogram2Dtgs
		
	// Lectura de proyecciones y reconstrucción, depende del tipo de dato de entrada:
	if(projectionType.compare("Sinogram2Dtgs")==0)
	{
	  // Sinograma 2D para TGS. Debe incluir los parámetros descriptos en la parte superior. Los leo,
	  // y si no están salgo porque son necesarios para la reconstrucción.
	  // "diameter_of_fov (in mm)"
	  if((errorCode=parametersFile_read((char*)parameterFileName.c_str(), (char*)"generateSystemMatrix", (char*)"diameter_of_fov (in mm)", (char*)returnValue, (char*)errorMessage)) != 0)
	  {
		  // Hubo un error. Salgo del comando.
		  if(errorCode == PMF_KEY_NOT_FOUND)
		  {
			cout<<"No se encontró el parámetro ""diameter_of_fov (in mm)"", el cual es obligatorio "
			"para el tipo de dato Sinogram2Dtgs."<<endl;
			return -1;
		  }
		  else
		  {
			cout<<"Error "<<errorCode<<" en el archivo de parámetros. Mirar la documentación de los códigos de errores."<<endl;
			return -1;
		  }
	  }
	  float diameterFov_mm = atoi(returnValue);
	  
	  // "distance_cristal_to_center_of_fov (in mm)"
	  if((errorCode=parametersFile_read((char*)parameterFileName.c_str(), (char*)"generateSystemMatrix", (char*)"distance_cristal_to_center_of_fov (in mm)", (char*)returnValue, (char*)errorMessage)) != 0)
	  {
		  // Hubo un error. Salgo del comando.
		  if(errorCode == PMF_KEY_NOT_FOUND)
		  {
			cout<<"No se encontró el parámetro ""distance_cristal_to_center_of_fov (in mm)"", el cual es obligatorio "
			"para el tipo de dato Sinogram2Dtgs."<<endl;
			return -1;
		  }
		  else
		  {
			cout<<"Error "<<errorCode<<" en el archivo de parámetros. Mirar la documentación de los códigos de errores."<<endl;
			return -1;
		  }
	  }
	  float distCrystalToCenterFov = atoi(returnValue);
	  
	  // "length_of_colimator (in mm)"
	  if((errorCode=parametersFile_read((char*)parameterFileName.c_str(), (char*)"generateSystemMatrix", (char*)"length_of_colimator (in mm)", (char*)returnValue, (char*)errorMessage)) != 0)
	  {
		  // Hubo un error. Salgo del comando.
		  if(errorCode == PMF_KEY_NOT_FOUND)
		  {
			cout<<"No se encontró el parámetro ""length_of_colimator (in mm)"", el cual es obligatorio "
			"para el tipo de dato Sinogram2Dtgs."<<endl;
			return -1;
		  }
		  else
		  {
			cout<<"Error "<<errorCode<<" en el archivo de parámetros. Mirar la documentación de los códigos de errores."<<endl;
			return -1;
		  }
	  }
	  float lengthColimator_mm = atoi(returnValue);
	  
	  // "diameter_of_colimator (in mm)"
	  if((errorCode=parametersFile_read((char*)parameterFileName.c_str(), (char*)"generateSystemMatrix", (char*)"diameter_of_colimator (in mm)", (char*)returnValue, (char*)errorMessage)) != 0)
	  {
		  // Hubo un error. Salgo del comando.
		  if(errorCode == PMF_KEY_NOT_FOUND)
		  {
			cout<<"No se encontró el parámetro ""diameter_of_colimator (in mm)"", el cual es obligatorio "
			"para el tipo de dato Sinogram2Dtgs."<<endl;
			return -1;
		  }
		  else
		  {
			cout<<"Error "<<errorCode<<" en el archivo de parámetros. Mirar la documentación de los códigos de errores."<<endl;
			return -1;
		  }
	  }
	  float widthHoleCollimator_mm = atoi(returnValue);
		
	  // Cargo la imagen de initial estimate, que esta en formato interfile, y de ella obtengo 
	  // los tamaños de la imagen.
	  image = new Image();
	  if(!(image->readFromInterfile((char*)imageFilename.c_str())))
	  {
		cout<<"Error al leer la imagen inicial: "<<image->getError()<<endl;
		return -1;
	  }
	
	  Sinogram2Dtgs* projection = new Sinogram2Dtgs();
	  projection->readFromInterfile(projectionFilename);
	  /// Ahora seteo los paramétros geométricos del sinograma:
/*	  projection->setGeometricParameters(diameterFov_mm/2, distCrystalToCenterFov, lengthColimator_mm, widthHoleCollimator_mm);*/
	  
	  /// Llamado de función que genera ls SMR.
	  
	}
	/*else if(inputType.compare("Sinogram3D")==0)
	{
	  // Sinograma 3D
	  
	}
	else if(inputType.compare("Michelogram")==0)
	{
	 
	}*/
	else
	{
	  cout<<"Tipo de dato de entrada no válido. Formatos válidos: ""Sinogram2d"""<<endl;
	  return -1;
	}
	
	
//	mlem->reconstructionImage->writeInterfile(sprintf("%s_end", outputPrefix.c_str()));
 
}
