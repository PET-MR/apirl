/**
	\file backproject.cpp
	\brief Archivo ejecutable para comando que realiza la backprojection o retroproyección.

	Este archivo genera el comando ejecutable para realizar la operación de backprojection. Para esto
	recibe como parámetro un archivo de configuración en formato interfile, donde se configuran todos
	los parámetros de la operación, que deberá incluir el tipo y nombre del dato de entrada, y además
	el tamaño de imagen de salida.
	\par Ejemplo de archivo de configuración de backprojection backproject.par:
	\code
		input file := 
		input type := Sinogram2D
	    Backproject Parameters :=
		; Projectors:
		backprojector := Siddon
	    output image := test_QP.h33
		output filename := test_QP
	    END :=
	\endcode
	\todo 
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2012.02.27
	\version 1.1.0
*/

#include <stdio.h>
#include <iostream>
#include <string.h>
#include <Michelogram.h>
#include <Mlem.h>
#include <Mlem2dTgs.h>
#include <Mlem2dTgsInSegment.h>
#include <MlemSinogram3d.h>
#include <ParametersFile.h>
#include <Projector.h>
#include <SiddonProjector.h>
#include <ConeOfResponseProjector.h>
#include <ConeOfResponseWithPenetrationProjector.h>
#include <SiddonProjectorWithAttenuation.h>
#include <ConeOfResponseProjectorWithAttenuation.h>
#include <ConeOfResponseWithPenetrationProjectorWithAttenuation.h>
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
	\brief Ejecutable que realiza la operación de retroproyección, para eso recibe como parámetro el nombre del archivo de configuración.
	
	Este comando realiza la operación de backprojection o retroproyección.
	Para esto recibe como argumento de entrada el nombre del archivo de configuración de parámetros de reconstrucción. Dicho argumento
	es obligatorio para poder ejecutar este comando, ya que en el se describen los parámetros necesarios para su ejecución.
	Se debe seleccionar un projector, por ahora los proyectores disponibles son:
	  -Siddon.
	  -ConeOfResponse. Parametro que requiere: "number_of_points_on_detector".
	  -ConeOfResponseWithPenetration. Parámetros que requiere: "number_of_points_on_detector",  
	  "number_of_points_on_collimator", "linear_attenuation_coeficient_cm".
	El archivo de parámetros tiene un formato similar a los archivos interfile, se diferencia en que la primera línea debe ser
	"Backprojection Parameter :=" y debe finalizar con un "END :=". Este formato está basado en el propuesto por STIR para configurar sus
	métodos de reconstrucción.
	Cada parámetro es ingresado en el archivo a través de keyword := value, siendo keyword el nombre del parámetros y value el valor
	que se le asigna. Hay campos obligatorios, y otros opcionales.
	Campos Obligatorios:
		- "input type" : tipo de entrada a reconstruir. Los valores posibles son: Sinogram2D, Sinogram2Dtgs, Sinogram3D y Michelogram.
		- "input file" : nombre del archivo header (*.hs) del sinograma a reconstruir en formato interfile.
		- "output image" : nombre de imagen existente en formato interfile, de donde se obtendrán los parámetros
		  (tamaño en píxeles, tamaño de píxel, etc) de la iamgen de salida. Se utiliza solo para lectura.
		- "output filename": nombre del archivo de imagen interfile de salida donde se guardará el resultado.
		- "backprojector" : proyector utilizado para la backprojection.	Campos Opcionales:

		- "attenuation image filename" : nombre de la iamgen interfile donde se encuentra el mapa de atenuación. El mapa
		  de atenuación tiene en cada voxel el coeficiente de atenuación lineal para la energía de aplicación.

	  Luego hay parámetros que son específicos a cada tipo de dato. Para el input type := Sinogram2Dtgs se deben cargar los 
	  siguientes parámetros:
		Sinogram2Dtgs Parameters :=		
		  diameter_of_fov (in mm) := 600
		  distance_cristal_to_center_of_fov (in mm) := 400
		  length_of_colimator (in mm) := 100
		  diameter_of_colimator (in mm) := 100
		  diameter_of_hole_colimator (in mm) := 20
		End Sinogram2Dtgs Parameters :=
	  Siendo:
		- "diameter_of_fov (in mm)" : diámetro del field of view.
		- "distance_cristal_to_center_of_fov (in mm)" : la distnacia en mm de la superficie del cristal al centro del fov.
		- "length_of_colimator (in mm)" : largo del colimador en mm.
		- "diameter_of_colimator (in mm)" : diámetro del total del colimador en mm.
		- "diameter_of_hole_colimator (in mm)" : diámetro del agujero del colimador en mm.
		
	  Para el input type := Sinogram2DtgsInSegment se deben cargar los siguientes parámetros:
		Sinogram2Dtgs Parameters :=		
		  diameter_of_fov (in mm) := 600
		  distance_cristal_to_center_of_fov (in mm) := 400
		  length_of_colimator (in mm) := 100
		  diameter_of_colimator (in mm) := 100
		  diameter_of_hole_colimator (in mm) := 20
		  width_of_segment (in mm) := 100
		End Sinogram2Dtgs Parameters :=
	  Siendo:
		- "diameter_of_fov (in mm)" : diámetro del field of view.
		- "distance_cristal_to_center_of_fov (in mm)" : la distnacia en mm de la superficie del cristal al centro del fov.
		- "length_of_colimator (in mm)" : largo del colimador en mm.
		- "diameter_of_colimator (in mm)" : diámetro del total del colimador en mm.
		- "diameter_of_hole_colimator (in mm)" : diámetro del agujero del colimador en mm.
		- "width_of_segment (in mm)" : altura del segmento adquirido.
	\par Ejemplo de Archivo de parámetro .par
	\code
	Backprojection Parameters :=
	; Ejemplo de archivo de configuración de reconstrucción MLEM.
	input type := Sinogram3D
	input file := test.hs
	; Projector:
	backprojector := Siddon
	; Attenuation Map (opcional):
	attenuation image filename := attenuationMap.hv
	output image := test_QP.h33
	output filename := outBackproject

	END :=
	\endcode

	@param argc Cantidad de argumentos de entrada
	@param argv Puntero a vector con los argumentos de entrada. El comando debe ejecutarse con el nombre del archivo de parámetros como argumento.
	@return 0 si no uhbo errores, 1  si falló la operación.
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\todo Hacer funciones por parámetros a configurar, es un choclazo sino.
	\date 2012.02.28
	\version 1.1.0
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
	float RFOV = FOV_radial / 2;
	float ZFOV = FOV_axial;
	char NombreRecon[50];
	string parameterFileName;	// string para el Nombre de Archivo de parámetros.
	string inputType;
	string inputFilename;	// string para el Nombre del archivo de header del sinograma.
	string outputImageFilename;	// string para el Nombre del archivo de imagen de salida.
	string outputFilename;	// string para el Nombre del archivo de imagen de salida.
	string strBackprojector;
	string attenMapFilename;
	Projector* backprojector;
	Image* outputImage;
	Image* attenuationImage;
	bool enableAttenuationCorrection = false;

	// Variables para sinogram2Dtgs y Sinogram2DtgsInSegment:
	float widthSegment_mm, diameterFov_mm, distCrystalToCenterFov, lengthColimator_mm, widthCollimator_mm, widthHoleCollimator_mm;
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
		cout << "El comando backprojection debe llamarse indicando el archivo de Parámetros de Reconstrucción: backprojection Param.par." << endl;
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
	strcpy(keyWords[0], "input type"); 
	strcpy(keyWords[1], "input file"); 
	strcpy(keyWords[2], "backprojector"); 
	strcpy(keyWords[3], "output image"); 
	strcpy(keyWords[4], "output filename"); 
	if((errorCode=parametersFile_readMultipleKeys((char*)parameterFileName.c_str(), (char*)"Backproject", (char**)keyWords, FIXED_KEYS, (char**)multipleReturnValue, errorMessage)) != 0)
	{
		// Hubo un error. Salgo del comando.
		cout<<"Error "<<errorCode<<" en el archivo de parámetros. Mirar la documentación de los códigos de errores."<<endl;
		return -1;
	}
	inputType.assign(multipleReturnValue[0]);
	inputFilename.assign(multipleReturnValue[1]);
	strBackprojector.assign(multipleReturnValue[2]);
	outputImageFilename.assign(multipleReturnValue[3]);
	outputFilename.assign(multipleReturnValue[4]);
	//outputFileName.assign(returnValue);
	
	/// Corrección por Atenuación.
	attenuationImage = new Image();
	// Es opcional, si está el mapa de atenuación se habilita:
	if((errorCode=parametersFile_read((char*)parameterFileName.c_str(), "Backproject", "attenuation image filename", returnValue, errorMessage)) != 0)
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
	  attenMapFilename.assign(returnValue);
	  enableAttenuationCorrection = true;
	  if(!attenuationImage->readFromInterfile((char*)attenMapFilename.c_str()))
	  {
		cout<<"Error al leer la imagen de mapa de atenuación." <<endl;
		return -1;
	  }
	}
	
	
	
	// Cargo la imagen de salida, y de ella obtengo 
	// los tamaños de la imagen.
	outputImage = new Image();
	if(!(outputImage->readFromInterfile((char*)outputImageFilename.c_str())))
	{
	  cout<<"Error al leer la imagen inicial: "<<outputImage->getError()<<endl;
	  return -1;
	}
	// La inicializo en cero:
	outputImage->fillConstant(0);
	
	if((strBackprojector.compare("Siddon") == 0)||(strBackprojector.compare("SiddonWithAttenuation") == 0))
	{
	  int numPointsOnDetector;
	  // En el Siddon por default tengo un único rayo, pero agrego un parámetro
	  // opcional para tener varias Lors por bin del sinograma.
	  // Debo leer el parámetro que tiene: "number_of_points_on_detector".
	  if((errorCode=parametersFile_read((char*)parameterFileName.c_str(), "Backproject", "number_of_points_on_detector", returnValue, errorMessage)) != 0)
	  {
		  // Hubo un error. Salgo del comando.
		  if(errorCode == PMF_KEY_NOT_FOUND)
		  {
			// No se definió se utiliza el valor por default:
			numPointsOnDetector = 1;
		  }
		  else
		  {
			cout<<"Error "<<errorCode<<" en el archivo de parámetros. Mirar la documentación de los códigos de errores."<<endl;
			return -1;
		  }
	  }
	  else
	  {
		numPointsOnDetector = atoi(returnValue);
	  }
	  if(strBackprojector.compare("Siddon") == 0)
	  {
		backprojector = (Projector*)new SiddonProjector(numPointsOnDetector);
	  }
	  else
	  {
		backprojector = (Projector*)new SiddonProjectorWithAttenuation(numPointsOnDetector, attenuationImage);
	  }
	}
	else if((strBackprojector.compare("ConeOfResponse") == 0)||(strBackprojector.compare("ConeOfResponseWithAttenuation") == 0))
	{
	  // Debo leer el parámetro que tiene: "number_of_points_on_detector".
	  if((errorCode=parametersFile_read((char*)parameterFileName.c_str(), "Backproject", "number_of_points_on_detector", returnValue, errorMessage)) != 0)
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
	  if(strBackprojector.compare("ConeOfResponse") == 0)
	  {
		backprojector = (Projector*)new ConeOfResponseProjector(numPointsOnDetector);
	  }
	  else
	  {
		backprojector = (Projector*)new ConeOfResponseProjectorWithAttenuation(numPointsOnDetector, attenuationImage);
	  }
	}
	else if(strBackprojector.compare("ConeOfResponseWithPenetration") == 0)
	{
	  // Debo leer los parámetros que tiene: "number_of_points_on_detector",  
	  // "number_of_points_on_collimator", "linear_attenuation_coeficient_cm".
	  if((errorCode=parametersFile_read((char*)parameterFileName.c_str(), "Backproject", "number_of_points_on_detector", returnValue, errorMessage)) != 0)
	  {
		  // Hubo un error. Salgo del comando.
		  if(errorCode == PMF_KEY_NOT_FOUND)
		  {
			cout<<"No se encontró el parámetro ""number_of_points_on_detector"", el cual es obligatorio "
			"para el proyector ConeOfResponseWithPenetration."<<endl;
			return -1;
		  }
		  else
		  {
			cout<<"Error "<<errorCode<<" en el archivo de parámetros. Mirar la documentación de los códigos de errores."<<endl;
			return -1;
		  }
	  }
	  int numPointsOnDetector = atoi(returnValue);
	  if((errorCode=parametersFile_read((char*)parameterFileName.c_str(), "Backproject", "number_of_points_on_collimator", returnValue, errorMessage)) != 0)
	  {
		  // Hubo un error. Salgo del comando.
		  if(errorCode == PMF_KEY_NOT_FOUND)
		  {
			cout<<"No se encontró el parámetro ""number_of_points_on_collimator"", el cual es obligatorio "
			"para el proyector ConeOfResponseWithPenetration."<<endl;
			return -1;
		  }
		  else
		  {
			cout<<"Error "<<errorCode<<" en el archivo de parámetros. Mirar la documentación de los códigos de errores."<<endl;
			return -1;
		  }
	  }
	  int numPointsOnCollimator = atoi(returnValue);
	  
	  if((errorCode=parametersFile_read((char*)parameterFileName.c_str(), "Backproject", "linear_attenuation_coeficient_cm", returnValue, errorMessage)) != 0)
	  {
		  // Hubo un error. Salgo del comando.
		  if(errorCode == PMF_KEY_NOT_FOUND)
		  {
			cout<<"No se encontró el parámetro ""linear_attenuation_coeficient_cm"", el cual es obligatorio "
			"para el proyector ConeOfResponseWithPenetration."<<endl;
			return -1;
		  }
		  else
		  {
			cout<<"Error "<<errorCode<<" en el archivo de parámetros. Mirar la documentación de los códigos de errores."<<endl;
			return -1;
		  }
	  }
	  float attenuationCoef_cm = atof(returnValue);
	  if(strBackprojector.compare("ConeOfResponseWithPenetration") == 0)
	  {
		backprojector = (Projector*)new ConeOfResponseWithPenetrationProjector(numPointsOnDetector, numPointsOnCollimator, attenuationCoef_cm);
	  }
	  else
	  {
		backprojector = (Projector*)new ConeOfResponseWithPenetrationProjectorWithAttenuation(numPointsOnDetector, numPointsOnCollimator, attenuationCoef_cm, attenuationImage);
	  }
	}
	else
	{
	  cout<<"Backprojector inválido."<<endl;
	  return -1;
	}
	
	cout<<"Iniciando backprojection:" << endl;
	
	// Lectura de proyecciones y reconstrucción, depende del tipo de dato de entrada. Sinogram2Dtgs y
	// Sinogram2DtgsInSegment tienen muchos parámetros en común:
	if((inputType.compare("Sinogram2Dtgs")==0)||(inputType.compare("Sinogram2DtgsInSegment")==0))
	{
	  // Sinograma 2D para TGS. Debe incluir los parámetros descriptos en la parte superior. Los leo,
	  // y si no están salgo porque son necesarios para la reconstrucción.
	  // "diameter_of_fov (in mm)"
	  if((errorCode=parametersFile_read((char*)parameterFileName.c_str(), "Backproject", "diameter_of_fov (in mm)", returnValue, errorMessage)) != 0)
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
	  diameterFov_mm = atoi(returnValue);
	  
	  // "distance_cristal_to_center_of_fov (in mm)"
	  if((errorCode=parametersFile_read((char*)parameterFileName.c_str(), "Backproject", "distance_cristal_to_center_of_fov (in mm)", returnValue, errorMessage)) != 0)
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
	  distCrystalToCenterFov = atoi(returnValue);
	  
	  // "length_of_colimator (in mm)"
	  if((errorCode=parametersFile_read((char*)parameterFileName.c_str(), "Backproject", "length_of_colimator (in mm)", returnValue, errorMessage)) != 0)
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
	  lengthColimator_mm = atoi(returnValue);
	  
	  // "diameter_of_colimator (in mm)"
	  if((errorCode=parametersFile_read((char*)parameterFileName.c_str(), "Backproject", "diameter_of_colimator (in mm)", returnValue, errorMessage)) != 0)
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
	  widthCollimator_mm = atoi(returnValue);
	  
	  // "diameter_of_colimator (in mm)"
	  if((errorCode=parametersFile_read((char*)parameterFileName.c_str(), "Backproject", "diameter_of_hole_colimator (in mm)", returnValue, errorMessage)) != 0)
	  {
		  // Hubo un error. Salgo del comando.
		  if(errorCode == PMF_KEY_NOT_FOUND)
		  {
			cout<<"No se encontró el parámetro ""diameter_of_hole_colimator (in mm)"", el cual es obligatorio "
			"para el tipo de dato Sinogram2Dtgs."<<endl;
			return -1;
		  }
		  else
		  {
			cout<<"Error "<<errorCode<<" en el archivo de parámetros. Mirar la documentación de los códigos de errores."<<endl;
			return -1;
		  }
	  }
	  widthHoleCollimator_mm = atoi(returnValue);
	}
	// Sección solo valída para sinogram2Dtgs
	if(inputType.compare("Sinogram2Dtgs")==0)
	{
	  Sinogram2Dtgs* inputProjection = new Sinogram2Dtgs();
	  if(!inputProjection->readFromInterfile(inputFilename))
	  {
		cout<<"Error al leer Sinogram2Dtgs en formato interfile." << endl;
		return -1;
	  }
	  /// Ahora seteo los paramétros geométricos del sinograma:
	  inputProjection->setGeometricParameters(diameterFov_mm/2, distCrystalToCenterFov, lengthColimator_mm, widthCollimator_mm,  widthHoleCollimator_mm);
	  backprojector->Backproject(inputProjection, outputImage);
	  
	}
	// Sección solo valída para sinogram2Dtgs
	else if(inputType.compare("Sinogram2DtgsInSegment")==0)
	{
	  // Para este tipo de sinograma tengo un parámetro más "width_of_segment (in mm)":
	  // "diameter_of_colimator (in mm)"
	  if((errorCode=parametersFile_read((char*)parameterFileName.c_str(), "Backproject", "width_of_segment (in mm)", returnValue, errorMessage)) != 0)
	  {
		  // Hubo un error. Salgo del comando.
		  if(errorCode == PMF_KEY_NOT_FOUND)
		  {
			cout<<"No se encontró el parámetro ""width_of_segment (in mm)"", el cual es obligatorio "
			"para el tipo de dato Sinogram2DtgsInSegment."<<endl;
			return -1;
		  }
		  else
		  {
			cout<<"Error "<<errorCode<<" en el archivo de parámetros. Mirar la documentación de los códigos de errores."<<endl;
			return -1;
		  }
	  }
	  float widthSegment_mm = atoi(returnValue);
	  
	  Sinogram2DtgsInSegment* inputProjection = new Sinogram2DtgsInSegment();
	  if(!inputProjection->readFromInterfile(inputFilename))
	  {
		cout<<"Error al leer Sinogram2DtgsInSegment en formato interfile." << endl;
		return -1;
	  }
	  /// Ahora seteo los paramétros geométricos del sinograma:
	  inputProjection->setGeometricParameters(diameterFov_mm/2, distCrystalToCenterFov, lengthColimator_mm, widthCollimator_mm,  widthHoleCollimator_mm, widthSegment_mm);
	  
	  backprojector->Backproject(inputProjection, outputImage);
	  
	}
	else if(inputType.compare("Sinogram3D")==0)
	{
	  // Sinograma 3D
	  Sinogram3D* inputProjection = new Sinogram3D((char*)inputFilename.c_str());
	  backprojector->Backproject(inputProjection, outputImage);
	}
	else if(inputType.compare("Michelogram")==0)
	{
	  // Sinograma 3D
	 // inputProjection = new Sinogram3D((char*)inputFilename.c_str());
	 /* SizeMichelogram sizeMichelogram;
	  sizeMichelogram.NProj = inputSinogram3D->NProj;
	  sizeMichelogram.NR = inputSinogram3D->NR;
	  sizeMichelogram.NZ = inputSinogram3D->NRings;
	  sizeMichelogram.RFOV = inputSinogram3D->RFOV;
	  sizeMichelogram.ZFOV = inputSinogram3D->ZFOV;
	  inputProjection = new Michelogram(sizeMichelogram);*/
	  //inputProjection->ReadDataFromSinogram3D(inputSinogram3D);
	  /*backprojector->Backproject(inputProjection, outputImage);
	  outputImage->writeInterfile(outputFilename.c_str());*/
	}
	else
	{
	  cout<<"Tipo de dato de entrada no válido. Formatos válidos: ""Sinogram2d"", ""Sinogram3D"", ""Michelogram"""<<endl;
	  return -1;
	}
	outputImage->writeInterfile((char*)outputFilename.c_str());
	cout<<"Operación de backprojection finalizada." << endl;
//	mlem->reconstructionImage->writeInterfile(sprintf("%s_end", outputPrefix.c_str()));
 
}
