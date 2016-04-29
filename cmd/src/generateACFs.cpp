/**
	\file generateACFs.cpp
	\brief Archivo ejecutable para comando de proyección.

	Este archivo genera el comando ejecutable para la creación de los attenuation correction
	factors a partir de una imagen. Se calcula siempre con el projector siddon ya que da el largo
	sobre el píxel. La imagen debe tener los coefcientes de atenuación lineal en 1/cm en cada píxel.
	\par Ejemplo de archivo de configuración de proyección projection.par:
	\code
	    input file := image.h33
		output type := Sinogram2D
	    Project Parameters :=
		output projection := test_QP.hs
		output filename := exampleproj
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
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <omp.h>
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
#include <readParameters.h>
#include <Sinogram3DArPet.h>
#include <Sinogram3DCylindricalPet.h>
#include <Sinograms2Din3DArPet.h>
#include <Sinograms2DinCylindrical3Dpet.h>
#include <Sinogram3DSiemensMmr.h>
#include <Sinograms2DinSiemensMmr.h>
#ifdef __USE_CUDA__
  #include <CuProjector.h>
  #include <CuProjectorInterface.h>
  #include <readCudaParameters.h>
#endif

#define FOV_axial 162
#define FOV_radial 582
#define FIXED_KEYS 4
using namespace std;
using	std::string;
/**
	\fn void main (int argc, char *argv[])
	\brief Ejecutable que realiza la proyección de una imagen, para eso recibe como parámetro el nombre del archivo de configuración de dicha operación.
	
	Este comando realiza la operación de proyección.
	Para esto recibe como argumento de entrada el nombre del archivo de configuración de parámetros de proyección. Dicho argumento
	es obligatorio para poder ejecutar este comando, ya que en el se describen los parámetros necesarios para su ejecución.
	Se debe seleccionar un projector, por ahora los proyectores disponibles son:
	  -Siddon.
	  -ConeOfResponse. Parametro que requiere: "number_of_points_on_detector".
	  -ConeOfResponseWithPenetration. Parámetros que requiere: "number_of_points_on_detector",  
	  "number_of_points_on_collimator", "linear_attenuation_coeficient_cm".
	El archivo de parámetros tiene un formato similar a los archivos interfile, se diferencia en que la primera línea debe ser
	"Projection Parameter :=" y debe finalizar con un "END :=". Este formato está basado en el propuesto por STIR para configurar sus
	métodos de reconstrucción.
	Cada parámetro es ingresado en el archivo a través de keyword := value, siendo keyword el nombre del parámetros y value el valor
	que se le asigna. Hay campos obligatorios, y otros opcionales.
	Campos Obligatorios:
		- "input file" : nombre del archivo header (*.h33) de la imagen a proyectar.
		- "output type" : tipo de proyección a generar. Los valores posibles son: Sinogram2D, Sinogram2Dtgs, Sinogram3D y Michelogram.
		- "output projection" : nombre de proyección o sinograma existente en formato interfile, de donde se obtendrán los parámetros
		  (cantidad de bins, ángulos, r, etc) de la proyección de salida. Se utiliza solo para lectura.
		- "output filename": nombre del archivo interfile de salida donde se guardará el resultado.
		- "projector" : proyector utilizado para la proyección.
		Campos Opcionales:
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
	Projection Parameters :=
	; Ejemplo de archivo de configuración de reconstrucción Projection.
	input file := image.h33
	output type := Sinogram2dTgs
	; Projector:
	projector := Siddon
	; Attenuation Map (opcional):
	attenuation image filename := attenuationMap.hv
	output projection := sinoExample.hs
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
	float rFov_mm = 0, axialFov_mm = 0, rScanner_mm = 0;
	char NombreRecon[50];
	string parameterFileName;	// string para el Nombre de Archivo de parámetros.
	
	string inputFilename;	// string para el Nombre del archivo de header del sinograma.
	Image* inputImage;
	string outputType;
	string sampleProjection;
	string outputFilename;	// string para el Nombre del archivo de imagen de salida.
	string attenMapFilename;
	string normProjFilename;
	string strForwardprojector;
	Projector* forwardprojector;
	Image* attenuationImage;
	bool enableAttenuationCorrection = false;
	
	#ifdef __USE_CUDA__
	  CuProjector* cuProjector;
	  CuProjectorInterface* cuProjectorInterface;
	  int gpuId;	// Id de la gpu a utilizar.
	  dim3 projectorBlockSize;	// Parámetros de cuda.
	#endif

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
		cout << "El comando Projection debe llamarse indicando el archivo de Parámetros de Reconstrucción: Projection Param.par." << endl;
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
	strcpy(keyWords[0], "input file"); 
	strcpy(keyWords[1], "output type");  
	strcpy(keyWords[2], "output filename");  
	strcpy(keyWords[3], "output projection");
	if((errorCode=parametersFile_readMultipleKeys((char*)parameterFileName.c_str(), (char*)"generateACFs", (char**)keyWords, FIXED_KEYS, (char**)multipleReturnValue, errorMessage)) != 0)
	{
	  // Hubo un error. Salgo del comando.
	  cout<<"Error "<<errorCode<<" en el archivo de parámetros. Mirar la documentación de los códigos de errores."<<endl;
	  return -1;
	}
	inputFilename.assign(multipleReturnValue[0]);
	outputType.assign(multipleReturnValue[1]);
	outputFilename.assign(multipleReturnValue[2]);
	sampleProjection.assign(multipleReturnValue[3]);
	
	// Leo el tamaño del FOV:
	if(getCylindricalScannerParameters(parameterFileName, "generateACFs", &rFov_mm, &axialFov_mm, &rScanner_mm))
	{
	  cout<<"Error al leer el tamaño del Fov." <<endl;
	  return -1;
	}
	
	// Cargo la imagen de initial estimate, que esta en formato interfile, y de ella obtengo 
	// los tamaños de la imagen.
	inputImage = new Image();
	if(!(inputImage->readFromInterfile((char*)inputFilename.c_str())))
	{
	  cout<<"Error al leer la imagen inicial: "<<inputImage->getError()<<endl;
	  return -1;
	}
	
	
	// Leer projector, que es opcional:
	strcpy(keyWords[0], "projector"); 
	if((errorCode=parametersFile_readMultipleKeys((char*)parameterFileName.c_str(), (char*)"generateACFs", (char**)keyWords, 1, (char**)multipleReturnValue, errorMessage)) != 0)
	{
	  // Hubo un error. Salgo del comando.
	  cout<<"Error "<<errorCode<<" en el archivo de parámetros. Mirar la documentación de los códigos de errores."<<endl;
	  return -1;
	}
	strForwardprojector.assign(multipleReturnValue[0]);
	
	// Inicializo el proyector a utilizar:
	if(strForwardprojector.compare("Siddon") == 0)
	{
	  forwardprojector = (Projector*)new SiddonProjector();
	}
	#ifdef __USE_CUDA__
	  else if(strForwardprojector.compare("CuSiddonProjector") == 0)
	  {
	    cuProjector = (CuProjector*)new CuSiddonProjector();
	    cuProjectorInterface = new CuProjectorInterface(cuProjector);
	    // Get size of kernels:
	    if(getProjectorBlockSize(parameterFileName, "generateACFs", &projectorBlockSize))
	    {
	      return -1;
	    }
	    if(getGpuId(parameterFileName, "generateACFs", &gpuId))
	    {
	      return -1;
	    }
	    // Set the configuration:
	    cuProjectorInterface->setGpuId(gpuId);
	    cuProjectorInterface->setProjectorBlockSizeConfig(projectorBlockSize);
	    forwardprojector = (Projector*)cuProjectorInterface;
	  }
	#endif
	else
	  forwardprojector = (Projector*)new SiddonProjector(); // Por defecto.

	// Paso la imagen a 1/mm, porque todas las distnacias están calculadas en mm en Siddon (verificarlo):
	SizeImage size = inputImage->getSize();
	for(int i = 0; i < size.nPixelsX; i++)
	{
	  for(int j = 0; j < size.nPixelsY; j++)
	  {
	    for(int k = 0; k < size.nPixelsZ; k++)
	    {
	      inputImage->setPixelValue(i,j,k,inputImage->getPixelValue(i,j,k)/10);
	    }
	  }
	}
	
	cout<<"Starting projection:" << endl;
	double startTime = omp_get_wtime();	
	// Ahora hago la proyección según el tipo de dato de entrada:
	if((outputType.compare("Sinogram2D")==0)||(outputType.compare("Sinogram2DinSiemensMmr")==0))
	{
	  Sinogram2D* outputProjection;
	  // Sinograma 2D de tipo genérico, este no tine ningún parámetro estra. Simplemente
	  // se lee la imagen del FOV.
	  if (outputType.compare("Sinogram2D")==0)
	  {
	    outputProjection = new Sinogram2DinCylindrical3Dpet((char*)sampleProjection.c_str(), rFov_mm, rScanner_mm);
	  }
	  else
	  {
	    outputProjection = new Sinogram2DinSiemensMmr((char*)sampleProjection.c_str());
	  }
	  
	  outputProjection->FillConstant(0);
	  forwardprojector->Project(inputImage, outputProjection);
	  
	  /// Compute atenuation correction factors
	  for(int j = 0; j < outputProjection->getNumProj(); j++)
	  {
		
	    for(int k = 0; k < outputProjection->getNumR(); k++)
	    {
	      /// Cada Sinograma 2D me represnta múltiples LORs, según la mínima y máxima diferencia entre anillos.
	      /// Por lo que cada bin me va a sumar cuentas en lors con distintos ejes axiales.
	      if(outputProjection->getSinogramBin(j,k) != 0)
	      {
		outputProjection->setSinogramBin(j,k,exp(outputProjection->getSinogramBin(j,k)));
	      }
	      else
	      {
		/// Si 0, fijar el acf en 1:
		outputProjection->setSinogramBin(j,k,1);
	      }
	    }
	  }    
	  outputProjection->writeInterfile(outputFilename);
	}
	else if((outputType.compare("Sinograms2D")==0)||(outputType.compare("Sinograms2DinCylindrical3Dpet")==0)||(outputType.compare("Sinograms2DinSiemensMmr")==0))
	{
	  Sinograms2DmultiSlice* outputProjection;
	  if((outputType.compare("Sinograms2D")==0)||(outputType.compare("Sinograms2DinCylindrical3Dpet")==0))
	  {
	    // Sinogramas 2D genérico (para scanner cilíndrico).
	    if(getCylindricalScannerParameters(parameterFileName, "generateACFs", &rFov_mm, &axialFov_mm, &rScanner_mm))
	    {
	      return -1;
	    }
	    outputProjection = new Sinograms2DinCylindrical3Dpet((char*)sampleProjection.c_str(), rFov_mm, axialFov_mm, rScanner_mm); 
	  }
	  else
	  {
	    outputProjection = new Sinograms2DinSiemensMmr((char*)sampleProjection.c_str()); 
	  }
	  outputProjection->FillConstant(0);
	  float x_mm, y_mm, z_mm;
	  // Recorro todos los slices y hago la proyección:
	  for(int i = 0; i < outputProjection->getNumSinograms(); i++)
	  {
	    Sinogram2D* sino2d = outputProjection->getSinogram2D(i);
	    sino2d->FillConstant(0);
	    float axialValueSino_mm = outputProjection->getAxialValue(i);
	    
	    // Obtengo el slice con las coordenadas del sinograma:
	    int j = 0; float distSliceToSino_mm;
	    inputImage->getPixelGeomCoord(0, 0, 0, &x_mm, &y_mm, &z_mm);
	    distSliceToSino_mm = abs(axialValueSino_mm - z_mm);
	    for(j = 1; j < inputImage->getSize().nPixelsZ; j++)
	    {
	      inputImage->getPixelGeomCoord(0, 0, j, &x_mm, &y_mm, &z_mm);
	      if (abs(axialValueSino_mm - z_mm)<distSliceToSino_mm)
		distSliceToSino_mm = abs(axialValueSino_mm - z_mm);
	      else
	      {
		j--;
		break;	// Si la distancia dejó de achicarse llegué al mínimo en el anterior.
	      }
	    }
	    // Si salió en el for porque llego a NpixelsZ, significa que el último slice es el mínimo, debo restarle uno para acceder al último slice:
	    if (j == inputImage->getSize().nPixelsZ)
	      j--;
	    Image* slice = inputImage->getSlice(j);
	    forwardprojector->Project(slice, sino2d);
	    // Copio el resultado del sino2d a al sinograma múltiple:
	    for(int indexAng = 0; indexAng < outputProjection->getNumProj(); indexAng++)
	    {
	      for(int indexR = 0; indexR < outputProjection->getNumR(); indexR++)
	      {
		outputProjection->getSinogram2D(i)->setSinogramBin(indexAng, indexR, sino2d->getSinogramBin(indexAng, indexR));
	      }
	    }
	    free(slice);
	  }
	  // Ahora lo paso a ACF, haciendo el log (supuestamente en la proyección obtuve la suma de los m*l:
	  for(int i = 0; i < outputProjection->getNumSinograms(); i++)
	  {
	    /// Cálculo de las coordenadas z del sinograma
	    for(int j = 0; j < outputProjection->getSinogram2D(i)->getNumProj(); j++)
	    {
		  
	      for(int k = 0; k < outputProjection->getSinogram2D(i)->getNumR(); k++)
	      {
		/// Cada Sinograma 2D me represnta múltiples LORs, según la mínima y máxima diferencia entre anillos.
		/// Por lo que cada bin me va a sumar cuentas en lors con distintos ejes axiales.
		if(outputProjection->getSinogram2D(i)->getSinogramBin(j,k) != 0)
		{
		  outputProjection->getSinogram2D(i)->setSinogramBin(j,k,exp(outputProjection->getSinogram2D(i)->getSinogramBin(j,k)));
		}
	      }
	    }
	  }
	  outputProjection->writeInterfile(outputFilename);
	  
	  // Aplico al sinograma de entrada los acf, esto es para probar y obtener un sinograma corregido directamente:
	  Sinograms2DinCylindrical3Dpet* inputProjection = new Sinograms2DinCylindrical3Dpet((char*)sampleProjection.c_str(),rFov_mm,axialFov_mm,rScanner_mm);
	  for(int i = 0; i < outputProjection->getNumSinograms(); i++)
	  {
	    /// Cálculo de las coordenadas z del sinograma
	    for(int j = 0; j < outputProjection->getSinogram2D(i)->getNumProj(); j++)
	    {
		  
	      for(int k = 0; k < outputProjection->getSinogram2D(i)->getNumR(); k++)
	      {
		/// Multiplico bin a bin para correfir por atenuación:
		inputProjection->getSinogram2D(i)->setSinogramBin(j,k,outputProjection->getSinogram2D(i)->getSinogramBin(j,k)*
		  inputProjection->getSinogram2D(i)->getSinogramBin(j,k));
	      }
	    }
	  }
	  sprintf(returnValue, "%s_attCorrected", outputFilename.c_str());
	  outputFilename.assign(returnValue);
	  inputProjection->writeInterfile(outputFilename);
	}
	else if(outputType.compare("Sinograms2Din3DArPet")==0)
	{
	  // Sinogramas 2D del ArPet. Un sinograma por cada slice o anillo.
	  // Leo el tamaño del FOV:
	  float blindDistance_mm; int minRingDiff;
	  if(getArPetParameters(parameterFileName, "Projection", &rFov_mm, &axialFov_mm, &blindDistance_mm, &minRingDiff))
	  {
	    return -1;
	  }
	  Sinograms2Din3DArPet* outputProjection = new Sinograms2Din3DArPet((char*)sampleProjection.c_str(), rFov_mm, axialFov_mm); 
	  outputProjection->FillConstant(0);
	  float x_mm, y_mm, z_mm;
	  // Recorro todos los slices y hago la proyección:
	  for(int i = 0; i < outputProjection->getNumSinograms(); i++)
	  {
	    Sinogram2D* sino2d = outputProjection->getSinogram2D(i);
	    sino2d->FillConstant(0);
	    float axialValueSino_mm = outputProjection->getAxialValue(i);
	    
	    // Obtengo el slice con las coordenadas del sinograma:
	    int j = 0; float distSliceToSino_mm;
	    inputImage->getPixelGeomCoord(0, 0, 0, &x_mm, &y_mm, &z_mm);
	    distSliceToSino_mm = abs(axialValueSino_mm - z_mm);
	    for(j = 1; j < inputImage->getSize().nPixelsZ; j++)
	    {
	      inputImage->getPixelGeomCoord(0, 0, j, &x_mm, &y_mm, &z_mm);
	      if(abs(axialValueSino_mm - z_mm)<distSliceToSino_mm)
		distSliceToSino_mm = abs(axialValueSino_mm - z_mm);
	      else
	      {
		j--;
		break;	// Si la distancia dejó de achicarse llegué al mínimo en el anterior.
	      }
	    }
	    // Si salió en el for porque llego a NpixelsZ, significa que el último slice es el mínimo, debo restarle uno para acceder al último slice:
	    if (j == inputImage->getSize().nPixelsZ)
	      j--;
	    Image* slice = inputImage->getSlice(j);
	    forwardprojector->Project(slice, sino2d);
	    // Copio el resultado del sino2d a al sinograma múltiple:
	    for(int indexAng = 0; indexAng < outputProjection->getNumProj(); indexAng++)
	    {
	      for(int indexR = 0; indexR < outputProjection->getNumR(); indexR++)
	      {
		outputProjection->getSinogram2D(i)->setSinogramBin(indexAng, indexR, sino2d->getSinogramBin(indexAng, indexR));
	      }
	    }
	    free(slice);

	  }
	  outputProjection->writeInterfile(outputFilename);
	}
	else if((outputType.compare("Sinogram3D")==0)||(outputType.compare("Sinogram3DSiemensMmr")==0)||(outputType.compare("Sinogram3DArPet")==0))
	{
	  // Sinograma 3D
	  Sinogram3D* outputProjection;
	  if(outputType.compare("Sinogram3D")==0)
	    outputProjection = new Sinogram3DCylindricalPet((char*)sampleProjection.c_str(),rFov_mm,axialFov_mm,rScanner_mm);
	  else if(outputType.compare("Sinogram3DSiemensMmr")==0)
	    outputProjection = new Sinogram3DSiemensMmr((char*)sampleProjection.c_str());
   	  else if (outputType.compare("Sinogram3DArPet")==0)
	    outputProjection = new Sinogram3DArPet((char*)sampleProjection.c_str(),rFov_mm,axialFov_mm);
	  //Sinogram3D* inputProjection = new Sinogram3DCylindricalPet((char*)sampleProjection.c_str(),rFov_mm,axialFov_mm,rScanner_mm);
	  //outputProjection->setGeometryDim(rFov_mm,axialFov_mm,rScanner_mm);
	  forwardprojector->Project(inputImage, outputProjection);
	  // Ahora lo paso a ACF, haciendo el log (supuestamente en la proyección obtuve la suma de los m*l:
	  for(int i = 0; i < outputProjection->getNumSegments(); i++)
	  {
	    for(int j = 0; j < outputProjection->getSegment(i)->getNumSinograms(); j++)
	    {
	      /// Cálculo de las coordenadas z del sinograma
	      for(int k = 0; k < outputProjection->getSegment(i)->getSinogram2D(j)->getNumProj(); k++)
	      {
		    
		for(int l = 0; l < outputProjection->getSegment(i)->getSinogram2D(j)->getNumR(); l++)
		{
		  // Al sinograma debo hacer que tenga solo una configuración de anillos posibles, porque me itneresa tener los largos
		  // de atenuación por cada píxel. Como esto es complicado, simplemente promedio entre todos los anillos de cada LOR.
		  // Update 16/02/14: ahora el proyector no usa todo los numZ al menos que se lo configure. Por lo tanto, saco la normalizacion por numz.
		  outputProjection->getSegment(i)->getSinogram2D(j)->setSinogramBin(k,l,outputProjection->getSegment(i)->getSinogram2D(j)->getSinogramBin(k,l));
		  /// Cada Sinograma 2D me represnta múltiples LORs, según la mínima y máxima diferencia entre anillos.
		  /// Por lo que cada bin me va a sumar cuentas en lors con distintos ejes axiales.
		  if(outputProjection->getSegment(i)->getSinogram2D(j)->getSinogramBin(k,l) != 0)
		  {
		    outputProjection->getSegment(i)->getSinogram2D(j)->setSinogramBin(k,l,exp(outputProjection->getSegment(i)->getSinogram2D(j)->getSinogramBin(k,l)));
		  }
		  else
		  {
		    /// Si 0, fijar el acf en 1:
		    outputProjection->getSegment(i)->getSinogram2D(j)->setSinogramBin(k,l,1);
		  }
		}
	      }
	    }
	  }
	  outputProjection->writeInterfile(outputFilename);
	  
	  // Aplico al sinograma de entrada los acf, esto es para probar y obtener un sinograma corregido directamente:
	  Sinogram3D* inputProjection = new Sinogram3DCylindricalPet((char*)sampleProjection.c_str(),rFov_mm,axialFov_mm,rScanner_mm);
	  for(int i = 0; i < outputProjection->getNumSegments(); i++)
	  {
	    for(int j = 0; j < outputProjection->getSegment(i)->getNumSinograms(); j++)
	    {
	      /// Cálculo de las coordenadas z del sinograma
	      for(int k = 0; k < outputProjection->getSegment(i)->getSinogram2D(j)->getNumProj(); k++)
	      {
		    
		for(int l = 0; l < outputProjection->getSegment(i)->getSinogram2D(j)->getNumR(); l++)
		{
		  // Al sinograma debo hacer que tenga solo una configuración de anillos posibles, porque me itneresa tener los largos
		  // de atenuación por cada píxel. Como esto es complicado, simplemente promedio entre todos los anillos de cada LOR.
		  inputProjection->getSegment(i)->getSinogram2D(j)->setSinogramBin(k,l,inputProjection->getSegment(i)->getSinogram2D(j)->getSinogramBin(k,l)*
		    outputProjection->getSegment(i)->getSinogram2D(j)->getSinogramBin(k,l));
		}
	      }
	    }
	  }
	  sprintf(returnValue, "%s_attCorrected", outputFilename.c_str());
	  outputFilename.assign(returnValue);
	  inputProjection->writeInterfile(outputFilename);
	}
	else if(outputType.compare("Michelogram")==0)
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
	}
	else
	{
	  cout<<"Tipo de dato de entrada no válido. Formatos válidos: ""Sinogram2d"", ""Sinogram3D"", ""Michelogram"""<<endl;
	  return -1;
	}
	double stopTime = omp_get_wtime();
	cout<<"Projection finished. Processing time: " << (stopTime-startTime) << "sec." << endl;
 
}
