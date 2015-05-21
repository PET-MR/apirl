/**
	\file cuMLEM.cpp
	\brief Archivo ejecutable para comando de reconstrucción con MLEM en GPU.

	Este archivo genera el comando ejecutable para reconstrucci�n con el algoritmo MLEM sobre GPU. Para esto
	recibe como par�metro un archivo de configuraci�n en formato interfile, donde se configuran todos
	los par�metros de reconstrucci�n, incluyendo en el mismo el dato de entrada a reconstruir. Est�
	implementado en CUDA. El comportamiento de este comando es igual al de MLEM para CPU.
	Prácticamente copié y pegué el de Mlem. Por lo que hay cosas que no están diponibles para gpu.
	\par Ejemplo de archivo de configuraci�n de reconstrucci�n reconMLEM.par:
	\code
	    MLEM Parameters :=
	    ; Ejemplo de archivo de configuraci�n de reconstrucci�n MLEM.
	    input file := test.hs
	    ; if the next parameter is disabled, 
	    ; the sensitivity will be computed
	    sensitivity filename := sens.hv
	    initial estimate := some_image
	    ; enable this when you read an initial estimate with negative data
	    enforce initial positivity condition := 0
	    output filename prefix := test_QP
	    number of iterations := 24
	    save estimates at subiteration intervals := 12
	    END :=
	\endcode
	\todo Agregar para guardar los valores de ML de cada iteraci�n y que los guarde en un txt.
	      Se deber�a quitar el input type cuando se tenga una clase projection que contenga todos los tipos
	      de datos de entrada.
	      Como necesito generar resultados r�pidos la implementaci�n la hago para sinogramas 3D del tipo del
	      PET GE Advance. Luego extenderlo a sinogramas de distinto tama�o, para eso los sinogramas deben
	      tener un m�todo que permita leer el intefile propietario. Se puede hacerlo tranquilamente con la
	      medcon al igual que para las im�genes.
	\bug
	\warning
	\author Mart�n Belzunce (martin.sure@gmail.com)
	\date 2010.11.24
	\version 1.0.0
*/

#include <stdio.h>
#include <iostream>
#include <string.h>
#include <Michelogram.h>
#include <CuMlemSinogram3d.h>
#include <ParametersFile.h>
#include <Images.h>
#include <Geometry.h>
#include <string>
#include <CuProjector.h>
#include <CuSiddonProjector.h>
#include <Sinograms2DinCylindrical3Dpet.h>
#include <Sinograms2Din3DArPet.h>
#include <Sinogram3DArPet.h>
#include <Sinogram3DSiemensMmr.h>
#include <Sinogram3DCylindricalPet.h>
#include <ParametersFile.h>
#include <readParameters.h>
#include <readCudaParameters.h>

#define FOV_axial 162
#define FOV_radial 582
#define FIXED_KEYS 5
using namespace std;
using	std::string;

/**
	\fn void main (int argc, char *argv[])
	\brief Ejecutable que realiza la reconstrucci�n MLEM en GPU, para eso recibe como par�metro el nombre del archivo de configuraci�n de reconstrucci�n.
	
	Este comando realiza la reconstrucci�n de un sinograma con el m�todo Maximum Likelihood Expectation Maximization (MLEM) sobre CUDA(GPU).
	Para esto recibe como argumento de entrada el nombre del archivo de configuraci�n de par�metros de reconstrucci�n. Dicho argumento
	es obligatorio para poder ejecutar este comando, ya que en el se describen los par�metros necesarios para su ejecuci�n y el nombre
	y caracter�sticas del sinograma a reconstruir.
	Hasta el momento el comando ejecuta siempre la recnstrucci�n en GPU a trav�s de la reconstrucci�n por Michelorgama, si el tipo
	de dato de entrada es del tipo Sinogram3D la convierte a Michelograma.
	El archivo de par�metros tiene un formato similar a los archivos interfile, se diferencia en que la primera l�nea debe ser
	"MLEMParameter :=" y debe finalizar con un "END :=". Este formato est� basado en el propuesto por STIR para configurar sus
	m�todos de reconstrucci�n.
	Cada par�metro es ingresado en el archivo a trav�s de keyword := value, siendo keyword el nombre del par�metros y value el valor
	que se le asigna. Hay campos obligatorios, y otros opcionales.
	Campos Obligatorios:
		- "input type" : tipo de entrada a reconstruir. Los valores posibles son: Sinogram2D, Sinogram3D y Michelogram.
		- "input file" : nombre del archivo header (*.hs) del sinograma a reconstruir en formato interfile.
		- "output filename prefix" : prefijo para los nombres de archivo de salida (principalmente nombres de las im�genes).
		- "number of iterations" : n�mero de iteraciones a realizar.
		- "initial estimate" : nombre del archivo de header de la imagen que ser� utilizada como estimador inicial. Puede ser
							   una imagen constante con valores mayores a 0. Esta imagen inicial, es a su vez la que se utilizar�
							   para determinar las caracter�sticas de la imagen de salida; esto es, la cantidad de dimensiones, 
							   el tama�o en p�xeles, y las dimensiones de cada p�xel en cada eje.
	Campos Opcionales:
		- "enforce initial positivity condition" : habilita (1) o deshabilita (0) el forzado de p�xeles positivo en la imagen inicial. Esta condici�n
												   es necesaria para asegurar la convergencia en MLEM (o Least Squares?). Si se omite esta
												   entrada se considera que est� dehabilitada.
	    - "save estimates at iteration intervals" : indica cada cuantas iteraciones se desea guardar la imagen de salida (Por ejemplo,
													para guardar las imagenes de salida de todas las iteraciones, este par�metro debe
													valer 1). Si se omite, por default vale 0, o sea que no se guardan los resultados de
													las iteraciones, sino que solo la imagen final.
		- "sensitivity filename" : nombre del header de la imagen de snesibilidad utilziada en el algoritmo MLEM. Esta imagen es la resultante de la
								   proyecci�n de una imagen constante. Si se omite este par�metro se calcula antes de iniciar la reconstrucci�n.

	\par Ejemplo de Archivo de par�metro .par
	\code
	MLEMParameters :=
	; Ejemplo de archivo de configuraci�n de reconstrucci�n MLEM.
	input type := Sinogram3D
	input file := test.hs
	; if the next parameter is disabled, 
	; the sensitivity will be computed
	sensitivity filename:= sens.hv
	initial estimate:= some_image
	; enable this when you read an initial estimate with negative data
	enforce initial positivity condition:=0
	output filename prefix := test_QP
	number of iterations := 24
	save estimates at subiteration intervals:= 12


	END :=
	\endcode

	@param argc Cantidad de argumentos de entrada
	@param argv Puntero a vector con los argumentos de entrada. El comando debe ejecutarse con el nombre del archivo de par�metros como argumento.
	@return 0 si no uhbo errores, 1  si fall� la operaci�n.
	\author Mart�n Belzunce (martin.sure@gmail.com)
	\date 2010.11.24
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
  string acfFilename, estimatedRandomsFilename, estimatedScatterFilename, normFilename;
  CuProjector* forwardprojector;
  CuProjector* backprojector;
  int saveIterationInterval;
  bool saveIntermediateData = false, bSensitivityFromFile = 0;
  Image* initialEstimate;
  unsigned int nIterations = 0;	// Número de iteraciones.
  CuMlemSinogram3d* mlem;	// Objeto mlem con el que haremos la reconstrucción. 
  int gpuId;	// Id de la gpu a utilizar.
  dim3 projectorBlockSize, backprojectorBlockSize, updateBlockSize;	// Parámetros de cuda.
  
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
    cout << "El comando MLEM debe llamarse indicando el archivo de Parámetros de Reconstrucción: MLEM Param.par." << endl;
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
  if((errorCode=parametersFile_readMultipleKeys((char*)parameterFileName.c_str(), (char*)"MLEM", (char**)keyWords, FIXED_KEYS, (char**)multipleReturnValue, errorMessage)) != 0)
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
  //outputFileName.assign(returnValue);

  // Lectura de los parámetros opcionales de reconstrucción MLEM:
  // "enforce initial positivity condition"
  // "save estimates at iteration intervals"
  // "sensitivity filename"
	  
  // Llamo a la función que obtiene del archivo de configuración si se debe cargar la sensitivity image o calcularlo. En el
  // caso de necesitar cargarla también se necesita el nombre del archivo:
  if(getSensitivityFromFile (parameterFileName, "MLEM", &bSensitivityFromFile, &sensitivityFilename))
  {
    return -1;
  }
  
  // Llamo a la función que procesa el archivo de entrada y busca si se deben guardar datos intermedios y su intervalo:
  if(getSaveIntermidiateIntervals (parameterFileName, "MLEM", &saveIterationInterval, &saveIntermediateData))
  {
    return -1;
  }
 
  // Obtengo los nombres (solo los nombres!) del projecctor y backprojector
  if(getProjectorBackprojectorNames(parameterFileName, "MLEM", &strForwardprojector, &strBackprojector))
  {
    return -1;
  }
  
  /// Corrección por Atenuación.
  // Es opcional, si está el mapa de atenuación se habilita:
  if((errorCode=parametersFile_read((char*)parameterFileName.c_str(), (char*)"MLEM", (char*)"attenuation image filename", (char*)returnValue, (char*)errorMessage)) != 0)
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
  
  // Parámetros de Cuda:
  if(getProjectorBlockSize(parameterFileName, "MLEM", &projectorBlockSize))
  {
    return -1;
  }
  if(getBackprojectorBlockSize(parameterFileName, "MLEM", &backprojectorBlockSize))
  {
    return -1;
  }
  if(getPixelUpdateBlockSize(parameterFileName, "MLEM", &updateBlockSize))
  {
    return -1;
  }
  if(getGpuId(parameterFileName, "MLEM", &gpuId))
  {
    return -1;
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
  if(strForwardprojector.compare("CuSiddonProjector") == 0)
  {
    forwardprojector = (CuProjector*)new CuSiddonProjector();
  }
  else if(strForwardprojector.compare("RotationBasedProjector") == 0)
  {
    // Ojo que por ahora no chequea si es al configuración del proyector o el backprojector.
    // Agarra el primer parámetro que encuentra.
    /*RotationBasedProjector::InterpolationMethods interpMethod;
    if (getRotationBasedProjectorParameters(parameterFileName, "MLEM", &interpMethod))
      return -1;
    forwardprojector = (Projector*)new RotationBasedProjector(interpMethod);*/
  }
  else if(strForwardprojector.compare("ArPetProjector") == 0)
  {
    //forwardprojector = (Projector*) new ArPetProjector();
  }
  else
  {
    printf("No existe el proyector %s.\n", strForwardprojector.c_str());
    exit(1);
  }

  // Inicializo del backprojector a utilizar. 
  if(strBackprojector.compare("CuSiddonProjector") == 0)
  {
    backprojector = (CuProjector*)new CuSiddonProjector();
  }
  else if(strBackprojector.compare("RotationBasedProjector") == 0)
  {
    // Ojo que por ahora no chequea si es al configuración del proyector o el backprojector.
    // Agarra el primer parámetro que encuentra.
    /*RotationBasedProjector::InterpolationMethods interpMethod;
    if (getRotationBasedProjectorParameters(parameterFileName, "MLEM", &interpMethod))
      return -1;
    backprojector = (Projector*)new RotationBasedProjector(interpMethod);*/
  }
  else if(strBackprojector.compare("ArPetProjector") == 0)
  {
    //backprojector = (Projector*) new ArPetProjector();
  }
  else
  {
    printf("No existe el proyector %s.\n", strForwardprojector.c_str());
    exit(1);
  }
  
  // Configruo los kernels de los proyectores:
  
  
  // Pido los singoramas de corrección si es que están disponibles:
  if(getCorrectionSinogramNames(parameterFileName, "MLEM", &acfFilename, &estimatedRandomsFilename, &estimatedScatterFilename))
    return -1;
  // Idem para normalización:
  if(getNormalizationSinogramName(parameterFileName,  "MLEM",&normFilename))
    return -1;
  // Lectura de proyecciones y reconstrucción, depende del tipo de dato de entrada:
  if(inputType.compare("Sinogram2D")==0)
  {
    // Sinograma 2D
    /*Sinogram2D* inputProjection = new Sinogram2DinCylindrical3Dpet((char*)inputFilename.c_str(), radiusFov_mm, radiusScanner_mm);
    mlem = new Mlem2d(inputProjection, initialEstimate, "", outputPrefix, numIterations, saveIterationInterval, saveIntermediateData, bSensitivityFromFile, forwardprojector, backprojector);
    if(bSensitivityFromFile)
    {
	  mlem->setSensitivityFilename(sensitivityFilename);
    }
    mlem->enableSaveIntermediates(saveIntermediateData);*/
  }
  else if(inputType.compare("Sinogram2DArPet")==0)
  {

  }
  else if(inputType.compare("Sinograms2Din3DArPet")==0)
  {
    /*float blindDistance_mm; int minDetDiff;
    if (getArPetParameters(parameterFileName, "MLEM", &radiusFov_mm, &zFov_mm, &blindDistance_mm, &minDetDiff))
      return -1;
    Sinograms2Din3DArPet* inputProjection = new Sinograms2Din3DArPet(inputFilename, radiusFov_mm, zFov_mm);
    inputProjection->setLengthOfBlindArea(blindDistance_mm);
    inputProjection->setMinDiffDetectors(minDetDiff);
    mlem = new Mlem2dMultiple((Sinograms2DmultiSlice*)inputProjection, initialEstimate, "", outputPrefix, numIterations, saveIterationInterval, saveIntermediateData, bSensitivityFromFile, forwardprojector, backprojector);
    if(bSensitivityFromFile)
    {
	  mlem->setSensitivityFilename(sensitivityFilename);
    }
    
    
    mlem->enableSaveIntermediates(saveIntermediateData);*/
  }
  else if((inputType.compare("Sinograms2D")==0)||(inputType.compare("Sinograms2DinCylindrical3Dpet")==0))
  {
    // Este es el caso donde tengo un sinograma 2d por cada slice de adquisición. Se reconstruye cada uno de ellos de forma
    // independiente. Lo que se necesita es el radio del scanner, el radio del fov y el zFov.
    // Para este tipo de datos, en el archivo de mlem me tienen que haber cargados los datos del scanner cilíndrico.
    /*if (getCylindricalScannerParameters(parameterFileName, "MLEM", &radiusFov_mm, &zFov_mm, &radiusScanner_mm))
      return -1;
    Sinograms2DinCylindrical3Dpet* inputProjection = new Sinograms2DinCylindrical3Dpet(inputFilename, radiusFov_mm, zFov_mm, radiusScanner_mm);
    mlem = new Mlem2dMultiple(inputProjection, initialEstimate, "", outputPrefix, numIterations, saveIterationInterval, saveIntermediateData, bSensitivityFromFile, forwardprojector, backprojector);
    if(bSensitivityFromFile)
    {
	  mlem->setSensitivityFilename(sensitivityFilename);
    }*/
  }
  else if(inputType.compare("Sinogram3DArPet")==0)
  {
    // Zona muerta:
    /*float blindArea_mm = 0; int minDetDiff;
    // Sinograma 3D
    // Para este tipo de datos, en el archivo de mlem me tienen que haber cargados los datos del scanner cilíndrico.
    if (getArPetParameters(parameterFileName, "MLEM", &radiusFov_mm, &zFov_mm, &blindArea_mm, &minDetDiff))
      return -1;
    Sinogram3DArPet* inputProjection = new Sinogram3DArPet((char*)inputFilename.c_str(),radiusFov_mm, zFov_mm); 
    inputProjection->setLengthOfBlindArea(blindArea_mm);
    inputProjection->setMinDiffDetectors(minDetDiff);
    mlem = new MlemSinogram3d(inputProjection, initialEstimate, "", outputPrefix, numIterations, saveIterationInterval, saveIntermediateData, bSensitivityFromFile, forwardprojector, backprojector);
    if(bSensitivityFromFile)
    {
	  mlem->setSensitivityFilename(sensitivityFilename);
    }*/
  }
  else if(inputType.compare("Sinogram3D")==0)
  {
    // Sinograma 3D
    // Para este tipo de datos, en el archivo de mlem me tienen que haber cargados los datos del scanner cilíndrico.
    if (getCylindricalScannerParameters(parameterFileName, "MLEM", &radiusFov_mm, &zFov_mm, &radiusScanner_mm))
    {
      cout<<"Error "<<errorCode<<" en el archivo de parámetros. No se pudieron obtener los parámetros para Sinogram3D."<<endl;
      return -1;
    }
    Sinogram3D* inputProjection = new Sinogram3DCylindricalPet((char*)inputFilename.c_str(),radiusFov_mm, zFov_mm, radiusScanner_mm);
    mlem = new CuMlemSinogram3d(inputProjection, initialEstimate, "", outputPrefix, numIterations, saveIterationInterval, saveIntermediateData, bSensitivityFromFile, forwardprojector, backprojector);
    if(bSensitivityFromFile)
    {
      mlem->setSensitivityFilename(sensitivityFilename);
    }
  }
  else if(inputType.compare("Sinogram3DSiemensMmr")==0)
  {
    // Sinograma 3D
    Sinogram3D* inputProjection = new Sinogram3DSiemensMmr((char*)inputFilename.c_str());
    mlem = new CuMlemSinogram3d(inputProjection, initialEstimate, "", outputPrefix, numIterations, saveIterationInterval, saveIntermediateData, bSensitivityFromFile, forwardprojector, backprojector);
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
    cout<<"Tipo de dato de entrada no válido. Formatos válidos: ""Sinogram2d"", ""Sinogram3D"", ""Sinogram3DSiemensMmr"", ""Michelogram"""<<endl;
    return -1;
  }
  // La habilitación de la data intermedia la debo hacer acá porque no está implementada en el constructor:
  if(saveIntermediateData)
    mlem->enableSaveIntermediates(saveIntermediateData);
  // Verifico si hay correciones para realizar.
  // Cargo los sinogramas de corrección:
  if(acfFilename != "")
    mlem->setAcfProjection(acfFilename);
  if(estimatedRandomsFilename != "")
    mlem->setRandomCorrectionProjection(estimatedRandomsFilename);
  if(estimatedScatterFilename != "")
    mlem->setScatterCorrectionProjection(estimatedScatterFilename);
  // Aplico las correciones:
  mlem->correctInputSinogram();
  // Y normalización:
  if (normFilename != "")
  {
    mlem->setNormalizationFactorsProjection(normFilename);
  }
  // Reconstruyo:
  TipoProyector tipoProy;
  tipoProy = SIDDON_PROJ_TEXT_CYLINDRICAL_SCANNER;
  // Asigno las configuraciones de ejecución:
  mlem->setBackprojectorKernelConfig(&backprojectorBlockSize);
  mlem->setProjectorKernelConfig(&projectorBlockSize);
  mlem->setUpdatePixelKernelConfig(&updateBlockSize);
  mlem->Reconstruct(tipoProy, gpuId);
//	mlem->reconstructionImage->writeInterfile(sprintf("%s_end", outputPrefix.c_str()));
 
}
