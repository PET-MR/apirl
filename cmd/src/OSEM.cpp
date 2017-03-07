/**
	\file OSEM.cpp
	\brief Archivo ejecutable para comando de reconstrucción con OSEM.

	Este archivo genera el comando ejecutable para reconstrucción con el algoritmo OSEM. Para esto
	recibe como parámetro un archivo de configuración en formato interfile, donde se configuran todos
	los parámetros de reconstrucción, incluyendo en el mismo el dato de entrada a reconstruir.
	\par Ejemplo de archivo de configuración de reconstrucción reconOSEM.par:
	\code
	    MLEM Parameters :=
	    ; Ejemplo de archivo de configuración de reconstrucción MLEM.
	    input file := test.hs
	    number of subsets := 12
	    ; if the next parameter is disabled, 
	    ; the sensitivity will be computed
	    sensitivity filename := sens.hv
		; Projectors:
		forwardprojector := Siddon
		backprojector := Siddon
	    initial estimate := some_image
	    ; enable this when you read an initial estimate with negative data
	    enforce initial positivity condition := 0
	    output filename prefix := test_QP
	    number of iterations := 24
	    save estimates at iteration intervals := 12
	    END :=
	\endcode
	\todo 
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2010.08.30
	\version 1.1.0
*/

#include <stdio.h>
#include <iostream>
#include <string.h>
#include <Michelogram.h>
#include <Mlem.h>
#include <Mlem2d.h>
#include <Mlem2dMultiple.h>
#include <Mlem2dTgs.h>
#include <MlemSinogram3d.h>
#include <OsemSinogram3d.h>
#include <Sinograms2DinCylindrical3Dpet.h>
#include <Sinograms2Din3DArPet.h>
#include <Sinogram3DArPet.h>
#include <Sinogram3DCylindricalPet.h>
#include <Sinogram3DSiemensMmr.h>
#include <ParametersFile.h>
#include <Projector.h>
#include <SiddonProjector.h>
#include <RotationBasedProjector.h>
#include <ArPetProjector.h>
#include <ConeOfResponseProjector.h>
#include <ConeOfResponseWithPenetrationProjector.h>
#include <Images.h>
#include <Geometry.h>
#include <string>
#include <readParameters.h>

#define FOV_axial 160
#define FOV_radial 400
#define SCANNER_RADIUS	700
#define FIXED_KEYS 6
using namespace std;
using	std::string;

/**
	\fn void main (int argc, char *argv[])
	\brief Ejecutable que realiza la reconstrucción MLEM, para eso recibe como parámetro el nombre del archivo de configuración de reconstrucción.
	
	Este comando realiza la reconstrucción de un sinograma con el método Maximum Likelihood Expectation Maximization (MLEM).
	Para esto recibe como argumento de entrada el nombre del archivo de configuración de parámetros de reconstrucción. Dicho argumento
	es obligatorio para poder ejecutar este comando, ya que en el se describen los parámetros necesarios para su ejecución y el nombre
	y características del sinograma a reconstruir. Se debe seleccionar un projector, por ahora los proyectores disponibles son:
	  -Siddon.
	  -RotationBasedProjector. Parámetro que requiere: "interpolation_method", que puede tomar los valores "nearest", 
		  "bilinear" o "bicubic"
	  -ConeOfResponse. Parametro que requiere: "number_of_points_on_detector".
	  -ConeOfResponseWithPenetration. Parámetros que requiere: "number_of_points_on_detector",  
	  "number_of_points_on_collimator", "linear_attenuation_coeficient_cm".
	El archivo de parámetros tiene un formato similar a los archivos interfile, se diferencia en que la primera línea debe ser
	"MLEMParameter :=" y debe finalizar con un "END :=". Este formato está basado en el propuesto por STIR para configurar sus
	métodos de reconstrucción.
	Cada parámetro es ingresado en el archivo a través de keyword := value, siendo keyword el nombre del parámetros y value el valor
	que se le asigna. Hay campos obligatorios, y otros opcionales.
	Campos Obligatorios:
		- "input type" : tipo de entrada a reconstruir. Los valores posibles son: Sinogram2D, Sinogram2Dtgs, Sinogram3D, Sinogram3DCylindricalPet, Sinogram3DSiemensMmr y Michelogram.
		- "input file" : nombre del archivo header (*.hs) del sinograma a reconstruir en formato interfile.
		- "number of subsets": cantidad de subsets en la que se divide cada sinograma.
		- "output filename prefix" : prefijo para los nombres de archivo de salida (principalmente nombres de las imágenes).
		- "number of iterations" : número de iteraciones a realizar.
		- "initial estimate" : nombre del archivo de header de la imagen que será utilizada como estimador inicial. Puede ser
							   una imagen constante con valores mayores a 0. Esta imagen inicial, es a su vez la que se utilizará
							   para determinar las características de la imagen de salida; esto es, la cantidad de dimensiones, 
							   el tamaño en píxeles, y las dimensiones de cada píxel en cada eje.
		- "forwardprojector" :  proyector utilizado para la forwardprojection.
		- "backprojector" : proyector utilizado para la backprojection.
	Campos Opcionales:
		- "attenuation image filename" : nombre de la iamgen interfile donde se encuentra el mapa de atenuación. El mapa
		  de atenuación tiene en cada voxel el coeficiente de atenuación lineal para la energía de aplicación.
		- "enforce initial positivity condition" : habilita (1) o deshabilita (0) el forzado de píxeles positivo en la imagen inicial. Esta condición
												   es necesaria para asegurar la convergencia en MLEM (o Least Squares?). Si se omite esta
												   entrada se considera que está dehabilitada.
	    - "save estimates at iteration intervals" : indica cada cuantas iteraciones se desea guardar la imagen de salida (Por ejemplo,
													para guardar las imagenes de salida de todas las iteraciones, este parámetro debe
													valer 1). Si se omite, por default vale 0, o sea que no se guardan los resultados de
													las iteraciones, sino que solo la imagen final.
		- "save estimated projections and backprojected image" : guarda la proyección estimada, y la backprojected image del cociente medida/estimada
										 para aquellas iteraciones donde se guarda la imagen de salida. Se uso para poder visualizar una secuencia de
										 convergencia o debuggear.
										 O sea guarda la estimated projections para cierto save estimates at iteration intervals. Si
										 se omite queda deshabilitada.
		- "sensitivity filename" : nombre del header de la imagen de snesibilidad utilziada en el algoritmo MLEM. Esta imagen 
		es la resultante de la proyección de una imagen constante. Si se omite este parámetro se calcula antes de iniciar la 
		reconstrucción.

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
	\par Ejemplo de Archivo de parámetro .par
	\code
	MLEMParameters :=
	; Ejemplo de archivo de configuración de reconstrucción MLEM.
	input type := Sinogram3D
	input file := test.hs
	; if the next parameter is disabled, 
	; the sensitivity will be computed
	sensitivity filename:= sens.hv
	initial estimate:= some_image
	; Projectors:
	forwardprojector := Siddon
	backprojector := Siddon
	; Attenuation Map (opcional):
	attenuation image filename := attenuationMap.hv
	; enable this when you read an initial estimate with negative data
	enforce initial positivity condition:=0
	output filename prefix := test_QP
	number of iterations := 24
	save estimates at subiteration intervals:= 12


	END :=
	\endcode

	@param argc Cantidad de argumentos de entrada
	@param argv Puntero a vector con los argumentos de entrada. El comando debe ejecutarse con el nombre del archivo de parámetros como argumento.
	@return 0 si no uhbo errores, 1  si falló la operación.
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\todo Hacer funciones por parámetros a configurar, es un choclazo sino.
	\date 2010.09.03
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
  float radiusFov_mm, zFov_mm, radiusScanner_mm;
  char NombreRecon[50];
  string parameterFileName;	// string para el Nombre de Archivo de parámetros.
  string inputType;
  string inputFilename;	// string para el Nombre del archivo de header del sinograma.
  string outputPrefix;	// prefijo para los nombres de los archivos de salida.
  string initialEstimateFilename;	// nombre del archivo de la imagen inicial.
  string sensitivityFilename;
  string strForwardprojector;
  string strBackprojector;
  string attenMapFilename;
  string multiplicativeFilename, additiveFilename;
  Projector* forwardprojector;
  Projector* backprojector;
  int saveIterationInterval;
  bool saveIntermediateData = false, bSensitivityFromFile = 0;
  Image* initialEstimate;
  unsigned int nIterations = 0;	// Número de iteraciones.
  int numberOfSubsets = 0;
  Mlem* mlem;	// Objeto mlem con el que haremos la reconstrucción. 
  
  // Asigno la memoria para los punteros dobles, para el array de strings.
  keyWords = (char**)malloc(sizeof(*keyWords)*FIXED_KEYS);
  multipleReturnValue = (char**)malloc(sizeof(*multipleReturnValue)*FIXED_KEYS);
  for(int j = 0; j < FIXED_KEYS; j++)
  {
    keyWords[j] = (char*) malloc(sizeof(char)*MAX_KEY_LENGTH);
    multipleReturnValue[j] = (char*) malloc(sizeof(char)*MAX_KEY_LENGTH);
  }
  //mlem = new MLEM();
  
  // Verificación de que se llamo al comando con el nombre de archivo de parámetros como argumento.
  if(argc != 2)
  {
    cout << "El comando OSEM debe llamarse indicando el archivo de Parámetros de Reconstrucción: OSEM Param.par." << endl;
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
  strcpy(keyWords[2], "initial estimate"); 
  strcpy(keyWords[3], "output filename prefix"); 
  strcpy(keyWords[4], "number of iterations"); 
  strcpy(keyWords[5], "number of subsets"); 
  if((errorCode=parametersFile_readMultipleKeys((char*)parameterFileName.c_str(), (char*)"OSEM", (char**)keyWords, FIXED_KEYS, (char**)multipleReturnValue, errorMessage)) != 0)
  {
    // Hubo un error. Salgo del comando.
    cout<<"Error "<<errorCode<<" en el archivo de parámetros. Mirar la documentación de los códigos de errores."<<endl;
    return -1;
  }
  inputType.assign(multipleReturnValue[0]);
  inputFilename.assign(multipleReturnValue[1]);
  initialEstimateFilename.assign(multipleReturnValue[2]);
  outputPrefix.assign(multipleReturnValue[3]);
  numIterations = atoi(multipleReturnValue[4]);
  numberOfSubsets = atoi(multipleReturnValue[5]);
  //outputFileName.assign(returnValue);

  // Lectura de los parámetros opcionales de reconstrucción MLEM:
  // "enforce initial positivity condition"
  // "save estimates at iteration intervals"
  // "sensitivity filename"
	  
  // Llamo a la función que obtiene del archivo de configuración si se debe cargar la sensitivity image o calcularlo. En el
  // caso de necesitar cargarla también se necesita el nombre del archivo:
  if(getSensitivityFromFile (parameterFileName, "OSEM", &bSensitivityFromFile, &sensitivityFilename))
  {
    return -1;
  }
  
  // Llamo a la función que procesa el archivo de entrada y busca si se deben guardar datos intermedios y su intervalo:
  if(getSaveIntermidiateIntervals (parameterFileName, "OSEM", &saveIterationInterval, &saveIntermediateData))
  {
    return -1;
  }
  
  // Obtengo los nombres (solo los nombres!) del projecctor y backprojector
  if(getProjectorBackprojectorNames(parameterFileName, "OSEM", &strForwardprojector, &strBackprojector))
  {
    return -1;
  }
  
  
  /// Corrección por Atenuación.
  // Es opcional, si está el mapa de atenuación se habilita:
  if((errorCode=parametersFile_read((char*)parameterFileName.c_str(), (char*)"OSEM", (char*)"attenuation image filename", (char*)returnValue, (char*)errorMessage)) != 0)
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
  }
  
  
  
  // Cargo la imagen de initial estimate, que esta en formato interfile, y de ella obtengo 
  // los tamaños de la imagen.
  initialEstimate = new Image();
  if(!(initialEstimate->readFromInterfile((char*)initialEstimateFilename.c_str())))
  {
    cout<<"Error al leer la imagen inicial: "<<initialEstimate->getError()<<endl;
    return -1;
  }
  
  // Inicializo el proyector a utilizar:
  if(strForwardprojector.compare("Siddon") == 0)
  {
    int numSamples, numAxialSamples;
	if(getSiddonProjectorParameters(parameterFileName, "OSEM", &numSamples, &numAxialSamples))
		return -1; // Return when is a big error, if the fields are not fouund they are already filled with the defaults.
	forwardprojector = (Projector*)new SiddonProjector(numSamples, numAxialSamples);
  }
  else if(strForwardprojector.compare("RotationBasedProjector") == 0)
  {
    // Ojo que por ahora no chequea si es al configuración del proyector o el backprojector.
    // Agarra el primer parámetro que encuentra.
	string strInterpMethod;
    if (getRotationBasedProjectorParameters(parameterFileName, "OSEM", &strInterpMethod))
      return -1;
    forwardprojector = (Projector*)new RotationBasedProjector(strInterpMethod);
  }
  else if(strForwardprojector.compare("ArPetProjector") == 0)
  {
    forwardprojector = (Projector*) new ArPetProjector();
  }

  // Inicializo del backprojector a utilizar. 
  if(strBackprojector.compare("Siddon") == 0)
  {
    int numSamples, numAxialSamples;
	if(getSiddonProjectorParameters(parameterFileName, "OSEM", &numSamples, &numAxialSamples))
		return -1; // Return when is a big error, if the fields are not fouund they are already filled with the defaults.
	backprojector = (Projector*)new SiddonProjector(numSamples, numAxialSamples);
  }
  else if(strBackprojector.compare("RotationBasedProjector") == 0)
  {
    // Ojo que por ahora no chequea si es al configuración del proyector o el backprojector.
    // Agarra el primer parámetro que encuentra.
    string strInterpMethod;
    if (getRotationBasedProjectorParameters(parameterFileName, "OSEM", &strInterpMethod))
      return -1;
    backprojector = (Projector*)new RotationBasedProjector(strInterpMethod);
  }
  else if(strBackprojector.compare("ArPetProjector") == 0)
  {
    backprojector = (Projector*) new ArPetProjector();
  }

  // Busco sinograma multiplicativo si se pasó alguno:
  if(getMultiplicativeSinogramName(parameterFileName,  "OSEM",&multiplicativeFilename))
    return -1;
  // Mismo con el additivo:
  if(getAdditiveSinogramName(parameterFileName,  "OSEM",&additiveFilename))
    return -1;
	
  // Lectura de proyecciones y reconstrucción, depende del tipo de dato de entrada:
  if(inputType.compare("Sinogram2D")==0)
  {
    // Sinograma 2D
    Sinogram2D* inputProjection = new Sinogram2DinCylindrical3Dpet((char*)inputFilename.c_str(), radiusFov_mm, radiusScanner_mm);
    mlem = new Mlem2d(inputProjection, initialEstimate, "", outputPrefix, numIterations, saveIterationInterval, saveIntermediateData, bSensitivityFromFile, forwardprojector, backprojector);
    if(bSensitivityFromFile)
    {
	  mlem->setSensitivityFilename(sensitivityFilename);
    }
    mlem->enableSaveIntermediates(saveIntermediateData);
  }
  else if(inputType.compare("Sinograms2Din3DArPet")==0)
  {
    float blindDistance_mm; int minDetDiff;
    if (getArPetParameters(parameterFileName, "OSEM", &radiusFov_mm, &zFov_mm, &blindDistance_mm, &minDetDiff))
      return -1;
    Sinograms2Din3DArPet* inputProjection = new Sinograms2Din3DArPet(inputFilename, radiusFov_mm, zFov_mm);
    inputProjection->setLengthOfBlindArea(blindDistance_mm);
    inputProjection->setMinDiffDetectors(minDetDiff);
    mlem = new Mlem2dMultiple((Sinograms2DinCylindrical3Dpet*)inputProjection, initialEstimate, "", outputPrefix, numIterations, saveIterationInterval, saveIntermediateData, bSensitivityFromFile, forwardprojector, backprojector);
    if(bSensitivityFromFile)
    {
	  mlem->setSensitivityFilename(sensitivityFilename);
    }
    mlem->enableSaveIntermediates(saveIntermediateData);
  }
  else if((inputType.compare("Sinograms2D")==0)||(inputType.compare("Sinograms2DinCylindrical3Dpet")==0))
  {
    // Este es el caso donde tengo un sinograma 2d por cada slice de adquisición. Se reconstruye cada uno de ellos de forma
    // independiente. Lo que se necesita es el radio del scanner, el radio del fov y el zFov.
    // Para este tipo de datos, en el archivo de mlem me tienen que haber cargados los datos del scanner cilíndrico.
    if (getCylindricalScannerParameters(parameterFileName, "OSEM", &radiusFov_mm, &zFov_mm, &radiusScanner_mm))
      return -1;
    Sinograms2DinCylindrical3Dpet* inputProjection = new Sinograms2DinCylindrical3Dpet(inputFilename, radiusFov_mm, zFov_mm, radiusScanner_mm);
    
    mlem = new Mlem2dMultiple(inputProjection, initialEstimate, "", outputPrefix, numIterations, saveIterationInterval, saveIntermediateData, bSensitivityFromFile, forwardprojector, backprojector);
    if(bSensitivityFromFile)
    {
	  mlem->setSensitivityFilename(sensitivityFilename);
    }
  }
  else if(inputType.compare("Sinogram3D")==0)
  {
    // Sinograma 3D
    // Para este tipo de datos, en el archivo de mlem me tienen que haber cargados los datos del scanner cilíndrico.
    if (getCylindricalScannerParameters(parameterFileName, "OSEM", &radiusFov_mm, &zFov_mm, &radiusScanner_mm))
      return -1;
    Sinogram3D* inputProjection = new Sinogram3DCylindricalPet((char*)inputFilename.c_str(),radiusFov_mm, zFov_mm, radiusScanner_mm);
    mlem = new OsemSinogram3d(inputProjection, initialEstimate, "", outputPrefix, numIterations, saveIterationInterval, saveIntermediateData, bSensitivityFromFile, forwardprojector, backprojector,numberOfSubsets);
    if(bSensitivityFromFile)
    {
      mlem->setSensitivityFilename(sensitivityFilename);
    }
  }
  else if(inputType.compare("Sinogram3DSiemensMmr")==0)
  {
    // Sinograma 3D
    Sinogram3D* inputProjection = new Sinogram3DSiemensMmr((char*)inputFilename.c_str());
    mlem = new OsemSinogram3d(inputProjection, initialEstimate, "", outputPrefix, numIterations, saveIterationInterval, saveIntermediateData, bSensitivityFromFile, forwardprojector, backprojector,numberOfSubsets);
    if(bSensitivityFromFile)
    {
      mlem->setSensitivityFilename(sensitivityFilename);
    }
  }
  else if(inputType.compare("Sinogram3DArPet")==0)
  {
    // Zona muerta:
    float blindArea_mm = 0; int minDetDiff;
    // Sinograma 3D
    // Para este tipo de datos, en el archivo de mlem me tienen que haber cargados los datos del scanner cilíndrico.
    if (getArPetParameters(parameterFileName, "OSEM", &radiusFov_mm, &zFov_mm, &blindArea_mm, &minDetDiff))
      return -1;
    Sinogram3DArPet* inputProjection = new Sinogram3DArPet((char*)inputFilename.c_str(),radiusFov_mm, zFov_mm); 
    inputProjection->setLengthOfBlindArea(blindArea_mm);
    inputProjection->setMinDiffDetectors(minDetDiff);
    mlem = new OsemSinogram3d((Sinogram3D*)inputProjection, initialEstimate, "", outputPrefix, numIterations, saveIterationInterval, saveIntermediateData, bSensitivityFromFile, forwardprojector, backprojector,numberOfSubsets);
    if(bSensitivityFromFile)
    {
	  mlem->setSensitivityFilename(sensitivityFilename);
    }
  }
  else if(inputType.compare("Michelogram")==0)
  {
    // Para este tipo de datos, en el archivo de mlem me tienen que haber cargados los datos del scanner cilíndrico.
    if (getCylindricalScannerParameters(parameterFileName, "MLEM", &radiusFov_mm, &zFov_mm, &radiusScanner_mm))
      return -1;
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
  }
  else
  {
    cout<<"Tipo de dato de entrada no válido. Formatos válidos: ""Sinogram2d"", ""Sinogram3D"", ""Michelogram"""<<endl;
    return -1;
  }
  // La habilitación de la data intermedia la debo hacer acá porque no está implementada en el constructor:
  if(saveIntermediateData)
    mlem->enableSaveIntermediates(saveIntermediateData);
  // Cargo los sinogramas de corrección:
  if(multiplicativeFilename != "")
    mlem->setMultiplicativeProjection(multiplicativeFilename);
  if(additiveFilename != "")
    mlem->setAdditiveProjection(additiveFilename);

  // Reconstruyo.
  mlem->Reconstruct();
//	mlem->reconstructionImage->writeInterfile(sprintf("%s_end", outputPrefix.c_str()));
 
}

