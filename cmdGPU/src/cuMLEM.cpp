/**
	\file cuMLEM.cpp
	\brief Archivo ejecutable para comando de reconstrucci�n con MLEM en GPU.

	Este archivo genera el comando ejecutable para reconstrucci�n con el algoritmo MLEM sobre GPU. Para esto
	recibe como par�metro un archivo de configuraci�n en formato interfile, donde se configuran todos
	los par�metros de reconstrucci�n, incluyendo en el mismo el dato de entrada a reconstruir. Est�
	implementado en CUDA. El comportamiento de este comando es igual al de MLEM para CPU.
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
#include <CudaMlem.h>
#include <ParametersFile.h>
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
	char errorMessage[300];	// string de error para la funci�n de lectura de archivo de par�metros.
	char returnValue[256];	// string en el que se recibe el valor de un keyword en la lectura del archivo de par�metros.
	char** keyWords;  // m�ltiples keywords para la funci�n de lectura de m�ltiples keys.
	char** multipleReturnValue; // array de strings para la funci�n de multples keys.
	int	errorCode;
	int nx, ny, nz, n_voxels, resolution, numIterations, file, Recon;
	unsigned int hTimer;
	const char* pathSensImage;
	double timerValue;
	unsigned int NProj = 192, NR = 192, NZ = 24;
	SizeImage MySizeVolume;
	float RFOV = FOV_radial / 2;
	float ZFOV = FOV_axial;
	char NombreRecon[50];
	string parameterFileName;	// string para el Nombre de Archivo de par�metros.
	string inputType;
	string inputFilename;	// string para el Nombre del archivo de header del sinograma.
	string outputPrefix;	// prefijo para los nombres de los archivos de salida.
	string initialEstimateFilename;	// nombre del archivo de la imagen inicial.
	string sensitivityFilename;
	int saveIterationInterval;
	Image* initialEstimate;
	unsigned int nIterations = 0;	// N�mero de iteraciones.
	CUDA_MLEM* cudaMlem;	// Objeto mlem con el que haremos la reconstrucci�n. 
	
	// Asigno la memoria para los punteros dobles, para el array de strings.
	keyWords = (char**)malloc(sizeof(*keyWords)*FIXED_KEYS);
	multipleReturnValue = (char**)malloc(sizeof(*multipleReturnValue)*FIXED_KEYS);
	for(int j = 0; j < FIXED_KEYS; j++)
	{
	  keyWords[j] = (char*) malloc(sizeof(char)*MAX_KEY_LENGTH);
	  multipleReturnValue[j] = (char*) malloc(sizeof(char)*MAX_KEY_LENGTH);
	}
	//mlem = new MLEM();
	
	// Verificaci�n de que se llamo al comando con el nombre de archivo de par�metros como argumento.
	if(argc != 2)
	{
		cout << "El comando MLEM debe llamarse indicando el archivo de Par�metros de Reconstrucci�n: MLEM Param.par." << endl;
		return -1;
	}
	// Los par�metros de reconstrucci�n son los correctos.
	// Se verifica que el archivo tenga la extensi�n .par.
	parameterFileName.assign(argv[1]);
	//strcpy(parameterFileName, argv[1]);
	if(parameterFileName.compare(parameterFileName.length()-4, 4, ".par"))
	{
		// El archivo de par�metro no tiene la extensi�n .par.
		cout<<"El archivo de par�metros no tiene la extensi�n .par."<<endl;
		return -1;
	}

	// Leo cada uno de los campos del archivo de par�metros. Para esto utilizo la funci�n parametersFile_readMultipleKeys
	// que  me permite leer m�ltiples keys en una �nico llamado a funci�n. Para esto busco los keywords que forman 
	// parte de los campos obligatorios, los opcionales los hago de a uno por vez.
	strcpy(keyWords[0], "input type"); 
	strcpy(keyWords[1], "input file"); 
	strcpy(keyWords[2], "initial estimate"); 
	strcpy(keyWords[3], "output filename prefix"); 
	strcpy(keyWords[4], "number of iterations"); 
	if((errorCode=parametersFile_readMultipleKeys((char*)parameterFileName.c_str(), (char*)"MLEM", (char**)keyWords, FIXED_KEYS, (char**)multipleReturnValue, errorMessage)) != 0)
	{
		// Hubo un error. Salgo del comando.
		cout<<"Error "<<errorCode<<" en el archivo de par�metros. Mirar la documentaci�n de los c�digos de errores."<<endl;
		return -1;
	}
	inputType.assign(multipleReturnValue[0]);
	inputFilename.assign(multipleReturnValue[1]);
	initialEstimateFilename.assign(multipleReturnValue[2]);
	outputPrefix.assign(multipleReturnValue[3]);
	numIterations = atoi(multipleReturnValue[4]);
	//outputFileName.assign(returnValue);

	// Lectura de los par�metros opcionales de reconstrucci�n MLEM:
	// "enforce initial positivity condition"
	// "save estimates at iteration intervals"
	// "sensitivity filename"
	if((errorCode=parametersFile_read((char*)parameterFileName.c_str(), "MLEM", "sensitivity filename", returnValue, errorMessage)) != 0)
	{
		// Hubo un error. Salgo del comando.
		// Si no encontr� el keyoword, est� bien porque era opcional, cualquier otro c�digo de error
		// signfica que hubo un error.
		if(errorCode == PMF_KEY_NOT_FOUND)
		{
		  // No est� la keyword, como era opcional se carga con su valor por default.
		  sensitivityFilename = "";
		}
		else
		{
		  cout<<"Error "<<errorCode<<" en el archivo de par�metros. Mirar la documentaci�n de los c�digos de errores."<<endl;
		  return -1;
		}
	}
	else
	{
	  sensitivityFilename.assign(returnValue);
	}
	// "save estimates at iteration intervals"
	if((errorCode=parametersFile_read((char*)parameterFileName.c_str(), "MLEM", "save estimates at iteration intervals", returnValue, errorMessage)) != 0)
	{
		// Hubo un error. Salgo del comando.
		// Si no encontr� el keyoword, est� bien porque era opcional, cualquier otro c�digo de error
		// signfica que hubo un error.
		if(errorCode == PMF_KEY_NOT_FOUND)
		{
		  // No est� la keyword, como era opcional se carga con su valor por default.
		  saveIterationInterval = 0;
		}
		else
		{
		  cout<<"Error "<<errorCode<<" en el archivo de par�metros. Mirar la documentaci�n de los c�digos de errores."<<endl;
		  return -1;
		}
	}
	else
	{
	  saveIterationInterval = atoi(returnValue);
	}
	// "enforce initial positivity condition"
	if((errorCode=parametersFile_read((char*)parameterFileName.c_str(), "MLEM", "enforce initial positivity condition", returnValue, errorMessage)) != 0)
	{
		// Hubo un error. Salgo del comando.
		// Si no encontr� el keyoword, est� bien porque era opcional, cualquier otro c�digo de error
		// signfica que hubo un error.
		if(errorCode == PMF_KEY_NOT_FOUND)
		{
		  // No est� la keyword, como era opcional se carga con su valor por default.
		}
		else
		{
		  cout<<"Error "<<errorCode<<" en el archivo de par�metros. Mirar la documentaci�n de los c�digos de errores."<<endl;
		  return -1;
		}
	}
	else
	{
	  
	}
	// Cargo la imagen de initial estimate, que esta en formato interfile, y de ella obtengo 
	// los tama�os de la imagen.
	initialEstimate = new Image();
	if(!(initialEstimate->readFromInterfile((char*)initialEstimateFilename.c_str())))
	{
	  cout<<"Error al leer la imagen inicial: "<<initialEstimate->getError()<<endl;
	  return -1;
	}
	// Lectura de proyecciones y reconstrucci�n, depende del tipo de dato de entrada:
	if(inputType.compare("Sinogram2D")==0)
	{
	  // Sinograma 2D
	  Sinogram2D* inputSinogram2D = new Sinogram2D();
	  cout<<"MLEM para sinograma 2D todav�a no."<<endl;
	}
	else if(inputType.compare("Sinogram3D")==0)
	{
	  // Sinograma 3D
	  Sinogram3D* inputSinogram3D = new Sinogram3D((char*)inputFilename.c_str());
	  SizeMichelogram sizeMichelogram;
	  sizeMichelogram.NProj = inputSinogram3D->getNumProj();
	  sizeMichelogram.NR = inputSinogram3D->getNumR();
	  sizeMichelogram.NZ = inputSinogram3D->getNumRings();
	  sizeMichelogram.RFOV = inputSinogram3D->getRadioFov_mm();
	  sizeMichelogram.ZFOV = inputSinogram3D->getAxialFoV_mm();
	  
	  Michelogram* inputMichelogram = new Michelogram(sizeMichelogram);
	  inputMichelogram->ReadDataFromSinogram3D(inputSinogram3D);
	  cudaMlem = new CUDA_MLEM(inputMichelogram, initialEstimate, outputPrefix, inputMichelogram->rScanner);
	  cudaMlem->setNumIterations(numIterations);
	  cudaMlem->setSaveIterationInterval(saveIterationInterval);
	  if(sensitivityFilename == "")
	  {
	    cudaMlem->setSensitivityImageFromFile(false);
	  }
	  else
	  {
	    cudaMlem->setSensitivityImageFromFile(true);
	    cudaMlem->setSensitivityFilename(sensitivityFilename);
	  }
	  
	  cudaMlem->ReconstruirEnGPU();
	}
	else if(inputType.compare("Michleogram")==0)
	{
	}
	else
	{
	  cout<<"Tipo de dato de entrada no v�lido. Formatos v�lidos: ""Sinogram2d"", ""Sinogram3D"", ""Michelogram"""<<endl;
	  return -1;
	}
//	mlem->reconstructionImage->writeInterfile(sprintf("%s_end", outputPrefix.c_str()));
 
}

/*
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <cutil.h>
#include <SparseMatrix.h>
#include <Michelogram.h>
#include <MLEM.h>
#include <Images.h>
#include <Geometry.h>
#include <CUDA_MLEM.h>
#define FOV_axial 16.2
#define FOV_radial 58.2
using namespace std;

void ComparacionSiddonPattern (SizeMichelogram MySizeMichelogram, SizeVolume MySizeVolume);
void ComparacionSensibilityImage (SizeMichelogram MySizeMichelogram, SizeVolume MySizeVolume);
void Comparacion_MLEM(Michelogram* MyMichelogram, SizeVolume MySizeVolume, unsigned int iterations, char* pathSensImage);
void Test_GPU(Michelogram* MyMichelogram, SizeVolume MySizeVolume, unsigned int iterations, char* pathSensImage);
void Testbench2D();
void VerificarLecturaSino3D (void);
void TestConversionSino3DtoMicho(char* FullInputPath, char* OutputPath);
void TestProjectionMicho(SizeVolume MySizeVolume, char* FullInputPath, char* OutputPath);
void ReconstruccionAATN (SizeVolume MySizeVolume, char* FullInputPath, char* OutputPath, char* Name, int CantIter);
void ReconstruccionAATN_GPU (SizeVolume MySizeVolume, char* FullInputPath, char* OutputPath, char* Name, int CantIter);

int main (int argc, char *argv[])
{
	int nx, ny, nz, n_voxels, resolution, iterations, file, Recon;
	unsigned int hTimer;
	const char* pathSensImage;
	double timerValue;
	unsigned int NProj = 192, NR = 192, NZ = 24;
	SizeVolume MySizeVolume;
	float RFOV = FOV_radial / 2;
	float ZFOV = FOV_axial;
	char NombreRecon[50];
	//MyMichelogram->ReadFromFile("C:\\Dokumente und Einstellungen\\Administrator\\Eigene Dateien\\Fellowship Martin Belzunce\\sino3DNema.s");
	CUT_SAFE_CALL(cutCreateTimer(&hTimer));

	// If the parameters were not passed through the arguments, we ask for them
	if(argc!=5)
	{
		cout << "Seleccione la reconstrucci�n a realizar..." << endl;
		cout << "1. AATN: GE_Advance" << endl;
		cout << "2. Siemens Biograph" << endl;
		if(argc==6)
			Recon = atoi(argv[1]);
		else
			cin >> Recon;
		

		switch(Recon)
		{
			case 1:
				/// Reconstrucci�n AATN
				{
					cout << "Please choose from the menu below the resolution..." << endl;
					cout << "1. 32x32x32" << endl;
					cout << "2. 128x128x47" << endl;
					cout << "3. 256x256x47" << endl;
					// Get and store their choice
					if(argc==6)
						resolution = atoi(argv[2]);
					else
						cin >> resolution;
					printf("%d\n",resolution);
					// Evaluate teh choice and call the selected functions
					switch (resolution) {
						case 1:
							 cout << "Selected resolution 32x32x32" << endl<<endl;
							 nx=32; ny=32; nz=32; MySizeVolume.Nx = 32; MySizeVolume.Ny = 32;MySizeVolume.Nz = 32;
							 break;
						case 2:
							 cout << "Selected resolution 128x128x47" << endl<<endl;
							 nx=128; ny=128; nz=47; MySizeVolume.Nx = 128; MySizeVolume.Ny = 128;MySizeVolume.Nz = 47;
							 break;
						case 3:
							 cout << "Selected resolution 256x256x47" << endl<<endl;
							 nx=256; ny=256; nz=47; MySizeVolume.Nx = 256; MySizeVolume.Ny = 256;MySizeVolume.Nz = 47;
							 break;
						default:
							 cout << "Wrong input" << endl;
					}
					n_voxels=nx*ny*nz;

					cout << "Please choose from the menu below the number of iterations.." << endl;
					cout << "1. 1  iteration" << endl;
					cout << "2. 5  iterations" << endl;
					cout << "3. 15 iterations" << endl;
					cout << "4. 42 iterations" << endl;
					// Get and store their choice
					if(argc==6)
						iterations = atoi(argv[3]);
					else
						cin >> iterations;

					// Evaluate teh choice and call the selected functions
					switch (iterations) {
						case 1:
							 cout << "Selected iteration 1" << endl<<endl;
							 iterations=1;
							 break;
						case 2:
							 cout << "Selected iteration 5" << endl<<endl;
							 iterations=5;
							 break;
						case 3:
							 cout << "Selected iteration 15" << endl<<endl;
							 iterations=15;
							 break;
						case 4:
							 cout << "Selected iteration 42" << endl<<endl;
							 iterations=42;
							 break;
						default:
							 cout << "Wrong input" << endl<<endl;
					}

					cout << "Please choose from the menu below the image that should be recontructed.." << endl;
					cout << "1. Cerebro" << endl;
					cout << "2. NEMA" << endl;
					cout << "3. NEMA_STIR" << endl;
					cout << "4. CILINDRO" << endl;
					cout << "5. CILINDRO_STIR" << endl;
					// Get and store their choice
					if(argc==6)
						file = atoi(argv[4]);
					else
						cin >> file;

					// Evaluate teh choice and call the selected functions
					switch (file) {
						case 1:
							 cout << "Sleeci�n de Sinograma 3D de Cerebro" << endl<<endl;
							 break;
						case 2:
							 cout << "Sleeci�n de Sinograma 3D de Fantoma NEMA" << endl<<endl;
						case 3:
							 cout << "Sleeci�n de Sinograma 3D de Fantoma NEMA_STIR" << endl<<endl;
							 break;
						 case 4:
							 cout << "Sleeci�n de Sinograma 3D de Fantoma CIL�NDRICO" << endl<<endl;
							 break;
						 case 5:
							 cout << "Sleeci�n de Sinograma 3D de Fantoma CIL�NDRICO_STIR" << endl<<endl;
							 break;
					} 
					
					char* fnam;
					char OutputPath[100];
					switch(file)
					{
						case 1:
							fnam ="E:\\Sinogramas\\SinogramasAATN\\sino3DCerebroCorrGeoNormAtt.s";
							printf("sino3DCerebroCorrGeoNormAtt.s\n"); 
							sprintf(OutputPath, "E:\\Reconstruction_Results\\AATN\\Cerebro\\%d_%d_%d", nx, ny, nz);
							MySizeVolume.RFOV = 15;
							MySizeVolume.ZFOV = 15.696;
							sprintf(NombreRecon, "Cerebro");
							//TestConversionSino3DtoMicho(fnam, "E:\\Reconstruction_Results\\AATN");
							//ReconstruccionAATN(MySizeVolume, fnam, OutputPath, "Cerebro", iterations);
							break;
						case 2:
							fnam="E:\\Sinogramas\\SinogramasAATN\\sino3DNema.s";
							printf("sino3DNema.s\n");
							sprintf(OutputPath, "E:\\Reconstruction_Results\\AATN\\NEMA\\%d_%d_%d", nx, ny, nz);
							MySizeVolume.RFOV = 25;
							MySizeVolume.ZFOV = 15.696;
							sprintf(NombreRecon, "NEMA");
							//ReconstruccionAATN(MySizeVolume, fnam, OutputPath, "NEMA", iterations);
							break;
						case 3:
							fnam="E:\\Sinogramas\\SinogramasAATN\\sinoNemaSTIR.s";
							printf("sino3DNema.s\n");
							sprintf(OutputPath, "E:\\Reconstruction_Results\\AATN\\NEMA_STIR\\%d_%d_%d", nx, ny, nz);
							MySizeVolume.RFOV = 25;
							MySizeVolume.ZFOV = 15.696;
							sprintf(NombreRecon, "NEMA_STIR");
							//ReconstruccionAATN(MySizeVolume, fnam, OutputPath, "NEMA", iterations);
							break;
						case 4:
							fnam="E:\\Sinogramas\\SinogramasAATN\\sinoCilindroCorregidoNormGeoAtt.s";
							printf("sinoCilindroCorregidoNormGeoAtt.s\n");
							sprintf(OutputPath, "E:\\Reconstruction_Results\\AATN\\CILINDRO\\%d_%d_%d", nx, ny, nz);
							MySizeVolume.RFOV = 35;
							MySizeVolume.ZFOV = 15.696;
							sprintf(NombreRecon, "CILINDRO");
							//ReconstruccionAATN(MySizeVolume, fnam, OutputPath, "NEMA", iterations);
							break;
						case 5:
							fnam="E:\\Sinogramas\\SinogramasAATN\\cilindro.s";
							printf("cilindro.s\n");
							sprintf(OutputPath, "E:\\Reconstruction_Results\\AATN\\CILINDRO_STIR\\%d_%d_%d", nx, ny, nz);
							MySizeVolume.RFOV = 25;
							MySizeVolume.ZFOV = 15.696;
							sprintf(NombreRecon, "CILINDRO_STIR");
							//ReconstruccionAATN(MySizeVolume, fnam, OutputPath, "NEMA", iterations);
							break;
						default:
							cout << "File number is invalid." << endl;
					}
					cout << "Seleccione sobre que plataforma quiere ejecutar al reconstrucci�n:" << endl;
					cout << "1. CPU" << endl;
					cout << "2. GPU" << endl;
					// Get and store their choice
					if(argc==6)
						file = atoi(argv[5]);
					else
						cin >> file;

					// Evaluate teh choice and call the selected functions
					switch (file) {
						case 1:
							 cout << "La reconstrucci�n se realizar� en el CPU." << endl<<endl;
							 break;
						case 2:
							cout << "La reconstrucci�n se realizar� en el GPU." << endl<<endl;
							break;
					}
					switch(file)
					{
						case 1:		
							//TestProjectionMicho(MySizeVolume, fnam, OutputPath);
							ReconstruccionAATN(MySizeVolume, fnam, OutputPath, NombreRecon, iterations);
							break;
						case 2:
							sprintf(OutputPath, "%s\\CUDA", OutputPath);
							ReconstruccionAATN_GPU(MySizeVolume, fnam, OutputPath, NombreRecon, iterations);
							break;
						default:
							cout << "File number is invalid." << endl;
					}
				}
				cin >> resolution;
				return 0;
				break;
			case 2:
				/// Reconstrucci�n Siemens Biogrpah
				cout << "Please choose from the menu below the resolution..." << endl;
				cout << "1. 32x32x32" << endl;
				cout << "2. 128x128x47" << endl;
				cout << "3. 256x256x47" << endl;
				// Get and store their choice
				cin >> resolution;
				printf("%d\n",resolution);
				// Evaluate teh choice and call the selected functions
				switch (resolution) {
					case 1:
						 cout << "Selected resolution 32x32x32" << endl<<endl;
						 nx=32; ny=32; nz=32; MySizeVolume.Nx = 32; MySizeVolume.Ny = 32;MySizeVolume.Nz = 32;
						 pathSensImage = "E:\\Final_Results\\Sensibility_Image_32_32_32.dat";
						 break;
					case 2:
						 cout << "Selected resolution 128x128x47" << endl<<endl;
						 nx=128; ny=128; nz=47; MySizeVolume.Nx = 128; MySizeVolume.Ny = 128;MySizeVolume.Nz = 47;
						 pathSensImage = "E:\\Final_Results\\Sensibility_Image_128_128_47.dat";
						 break;
					case 3:
						 cout << "Selected resolution 192x192x54" << endl<<endl;
						 nx=192; ny=192; nz=54; MySizeVolume.Nx = 192; MySizeVolume.Ny = 192;MySizeVolume.Nz = 54;
						 pathSensImage = "E:\\Final_Results\\Sensibility_Image_192_192_54.dat";
						 break;
					default:
						 cout << "Wrong input" << endl;
				}
				n_voxels=nx*ny*nz;

				cout << "Please choose from the menu below the number of iterations.." << endl;
				cout << "1. 1  iteration" << endl;
				cout << "2. 5  iterations" << endl;
				cout << "3. 15 iterations" << endl;
				cout << "4. 30 iterations" << endl;
				// Get and store their choice
				cin >> iterations;
				// Evaluate teh choice and call the selected functions
				switch (iterations) {
					case 1:
						 cout << "Selected iteration 1" << endl<<endl;
						 iterations=1;
						 break;
					case 2:
						 cout << "Selected iteration 5" << endl<<endl;
						 iterations=5;
						 break;
					case 3:
						 cout << "Selected iteration 15" << endl<<endl;
						 iterations=15;
						 break;
					case 4:
						 cout << "Selected iteration 30" << endl<<endl;
						 iterations=30;
						 break;
					default:
						 cout << "Wrong input" << endl<<endl;
				}

				cout << "Please choose from the menu below the image that should be recontructed.." << endl;
				cout << "1. Pig_SpheresOnly" << endl;
				cout << "2. Motion_Phantom_static.L" << endl;
				cout << "3. Patient_3" << endl;
				cout << "4. Patient_1" << endl;
				cout << "5. Gate_Simluation" << endl;
				cout << "6. Gate_Simluation_more_counts" << endl;
				// Get and store their choice
				cin >> file;
				// Evaluate teh choice and call the selected functions
				switch (file) {
					case 1:
						 cout << "Selected file Pig_SpheresOnly" << endl<<endl;
						 break;
					case 2:
						 cout << "Selected file Motion_Phantom_static.L" << endl<<endl;
						 break;
					case 3:
						 cout << "Selected file Patient_3" << endl<<endl;
						 break;
					case 4:
						 cout << "Selected file Patient_1" << endl<<endl;
						 break;
					case 5:
						cout << "Gate Simulation" << endl << endl;
						break;
					case 6:
						cout << "Gate Simulation more counts" << endl << endl;
						break;
					default:
						 cout << "Wrong input" << endl<<endl;
				} 
				
				const char* fnam;
				switch(file)
				{
						case 1:
							fnam ="E:\\Image_data_Biograph16\\Pig_SpheresOnly_18102006_corr.L";
							printf("Pig_SpheresOnly_18102006_corr.L\n"); 
							break;
						case 2:
							fnam="E:\\Image_data_Biograph16\\Motion_Phantom_static_29112006_corr.L";
							printf("Motion_Phantom_static.L\n");
							break;
						 case 3:
							fnam="E:\\Image_data_Biograph16\\Patient_3_08042005_corr.L";
							printf("Patient_1_08042005_corr.L\n"); 
							break;
						case 4:
							fnam="E:\\Image_data_Biograph16\\Patient_1_rest_30062005_corr.L";
							printf("Patient_3.L\n");
							break;
						case 5:
							fnam = "E:\\GATE_Michelograms\\Michelogram_192_192_10_10.dat";
							printf("Michelogram_192_192_10_10.dat");
							RFOV = 37;
							ZFOV = 30;
							NProj = 192;
							NR = 192;
							NZ =10;
							break;
						case 6:
							fnam = "E:\\GATE_Michelograms\\Michelogram_192_192_30_30.dat";
							printf("Michelogram_192_192_30_30.dat");
							RFOV = 37;
							ZFOV = 30;
							NProj = 192;
							NR = 192;
							NZ =30;
							break;
						default:
							cout << "File number is invalid." << endl;
					}
				return 0; break;
		}

		
	}
	
    /*

	
	MySizeVolume.RFOV = RFOV;
	MySizeVolume.ZFOV = ZFOV;
	SizeMichelogram MySizeMichelogram;
	MySizeMichelogram.NProj = NProj;
	MySizeMichelogram.NR = NR;
	MySizeMichelogram.NZ = NZ;
	MySizeMichelogram.RFOV = RFOV;
	MySizeMichelogram.ZFOV = ZFOV;
	*/

	
	//CORRO LOS DISTINTOS TESTS
	//ComparacionSiddonPattern (MySizeMichelogram, MySizeVolume);
	//ComparacionSensibilityImage (MySizeMichelogram, MySizeVolume);
	//Comparacion_MLEM(MyMichelogram, MySizeVolume, iterations, (char*)pathSensImage);
	//Testbench2D();
	//Test_GPU(MyMichelogram, MySizeVolume, iterations, (char*)pathSensImage);
	/*
	Volume *PointSource = new Volume(32,32,32);
	PointSource->Images2D[15]->Pixels[15*32+15] = 1;
	PointSource->Images2D[15]->Pixels[15*32+16] = 1;
	PointSource->Images2D[15]->Pixels[16*32+15] = 1;
	PointSource->Images2D[15]->Pixels[16*32+16] = 1;
	PointSource->Images2D[16]->Pixels[15*32+15] = 1;
	PointSource->Images2D[16]->Pixels[15*32+16] = 1;
	PointSource->Images2D[16]->Pixels[16*32+15] = 1;
	PointSource->Images2D[16]->Pixels[16*32+16] = 1;
	PointSource->SaveInFile("E:\\Final_Results\\PointSource_32_32_32.dat");
	Michelogram* PointSourceMichelogram = new Michelogram(192,192,24,1,1,FOV_radial/2, FOV_axial);
	Geometric3DProjectionV2 (PointSource, PointSourceMichelogram, RScanner);
	PointSourceMichelogram->SaveInFile("E:\\Final_Results\\PointSourceMichelogram_192_192_24.dat");
	MLEM(PointSourceMichelogram, PointSource, iterations, "E:\\Final_Results\\Sensibility_Image_32_32_32.dat");
	/////////////////////////////////////////////////////////////////////////*/
/*
	if (file < 5)
	{
		Michelogram* MyMichelogram = new Michelogram(192,192,24,1,1,FOV_radial/2,FOV_axial); // This scanner has 24 rings
		Volume* MyVolume = new Volume(nx,ny,nz);
		// Initialization of the Volume with a value different to zero
		MyVolume->FillConstant(0.2);
		MyMichelogram->ReadDataFromSiemensBiograph16((char*)fnam);
		MyMichelogram->SaveInFile("E:\\Reconstruction_Results\\FullMichelogram_float_192_192_24_24.dat");
//	Michelogram* MyMichelogram = new Michelogram(NProj,NR,NZ,1,1,RFOV,ZFOV); // This scanner has 24 rings
		MyMichelogram->ReadFromFile((char*) fnam);
}*/
	/// Prueba Sino 3D
	//VerificarLecturaSino3D ();
	/// Verifico todos los campos
	/*Sinogram3D* MiSino3D = new Sinogram3D("E:\\Sinogramas\\Sinograma GE\\sino3DNema.s");
	MySizeVolume.RFOV = 35;
	MySizeVolume.ZFOV = 15.696;
	Volume* MyVolume = new Volume(MySizeVolume);
	MyVolume->FillConstant(1.0);
	MySizeMichelogram.NProj = MiSino3D->NProj;
	MySizeMichelogram.NR = MiSino3D->NR;
	MySizeMichelogram.NZ = MiSino3D->NRings;
	MySizeMichelogram.RFOV = MiSino3D->RFOV;
	MySizeMichelogram.ZFOV = MiSino3D->ZFOV;
	Michelogram* MiMichelogram =  new Michelogram(MySizeMichelogram);
	MiMichelogram->ReadDataFromSinogram3D(MiSino3D);*/
	/// Reconstrucci�n CUDA
	//MLEM(MyMichelogram, MyVolume, iterations, "E:\\Final_Results\\Sensibility_Image_32_32_32.dat");
	/*cout << "Starting Martin GPU Online Reconstruction..." << endl;
	CUT_SAFE_CALL( cutResetTimer(hTimer) );
	CUT_SAFE_CALL( cutStartTimer(hTimer) );
	CUDA_MLEM(MiMichelogram, MyVolume, iterations, "E:\\Reconstruction_Results\\PruebaRecon2.0\\Sino3D_GE\\SensibilityVolumeMichelogram.dat","E:\\Reconstruction_Results\\PruebaRecon2.0\\Sino3D_GE\\CUDA_Michelogram");
	//CUDA_MLEM(MyMichelogram, MyVolume, iterations, "E:\\Reconstruction_Results\\CUDA2");
	CUT_SAFE_CALL(cutStopTimer(hTimer));
	timerValue = cutGetTimerValue(hTimer);
	cout << "Martin GPU Online Reconstruction finsihed." << endl;
	cout << "Reconstruction Time: " << timerValue / 1000 << "seg." << endl;*/
	//MiMichelogram->SaveInFile("E:\\Reconstruction_Results\\PruebaRecon2.0\\Sino3D_GE\\CUDA_Michelogram\\MicheFromSino3D.dat");
	//MiMichelogram->FillConstant(1.0);
	/*MLEM* PruebaMLEM = new MLEM(MiMichelogram, MyVolume, "E:\\Reconstruction_Results\\PruebaRecon2.0","Michelogram", 88.62/2);
	//PruebaMLEM->CalcularSensibility(MyVolume);
	//PruebaMLEM->SensibilityVolume->SaveInFile("E:\\Reconstruction_Results\\PruebaRecon2.0\\Sino3D_GE\\CUDA_Michelogram\\SensibilityVolume.dat");
	//PruebaMLEM->BackProjection(MiMichelogram, MyVolume);
	//MyVolume->SaveInFile("E:\\Reconstruction_Results\\PruebaRecon2.0\\Sino3D_GE\\CUDA_Michelogram\\BackprojectedVolume.dat");
	//PruebaMLEM->ForwardProjection(MyVolume, MiMichelogram);
	//MiMichelogram->SaveInFile("E:\\Reconstruction_Results\\PruebaRecon2.0\\Sino3D_GE\\CUDA_Michelogram\\ProjectedMicho.dat");
	PruebaMLEM->SensibilityFromFile = false;
	//sprintf(PruebaMLEM->PathSensibility, "E:\\Reconstruction_Results\\PruebaRecon2.0\\Sino3D_GE\\SensibilityVolumeMichelogram.dat");
	PruebaMLEM->CantIteraciones = 30;
	PruebaMLEM->GuardarImagenIteracion = true;
	PruebaMLEM->Reconstruir();*/
	//PruebaMLEM->CalcularSensibility(MyVolume);
	//PruebaMLEM->SensibilityVolume->SaveInFile("E:\\Reconstruction_Results\\PruebaRecon2.0\\Sino3D_GE\\SensibilityVolumeMichelogram.dat");
	/// Senisbility con Sino3D
	/*MLEM* PruebaMLEM = new MLEM(MiSino3D, MyVolume, "E:\\Reconstruction_Results\\PruebaRecon2.0","Prueba", 88.62/2);
	PruebaMLEM->ForwardProjection(MyVolume, MiSino3D);
	MiSino3D->SaveInFile("E:\\Reconstruction_Results\\PruebaRecon2.0\\Sino3D_GE\\ProjectionConstantVolume.dat");
	*/
	/*PruebaMLEM->CalcularSensibility(MyVolume);
	PruebaMLEM->SensibilityVolume->SaveInFile("E:\\Reconstruction_Results\\PruebaRecon2.0\\Sino3D_GE\\SensibilityVolume.dat");
	MiSino3D->~Sinogram3D();

	/// Sensibility con Michelogram
	Michelogram* MyMichelogram = new Michelogram(MySizeMichelogram);
	MyMichelogram->ReadDataFromSiemensBiograph16((char*)fnam);
	MyMichelogram->SaveInFile("E:\\Reconstruction_Results\\FullMichelogram_float_192_192_24_24.dat");
	PruebaMLEM = new MLEM(MyMichelogram, MyVolume, "E:\\Reconstruction_Results\\PruebaRecon2.0","Prueba", 41.35);
	PruebaMLEM->GuardarImagenIteracion = true;
	PruebaMLEM->CalcularSensibility(MyVolume);
	PruebaMLEM->SensibilityVolume->SaveInFile("E:\\Reconstruction_Results\\PruebaRecon2.0\\SensibilityVolume.dat");
	*/
	//PruebaMLEM->Reconstruir();

	/*////////////////GPU RDI RECONSTRUCTION//////////////////
	cout << "Starting  GPU Resolution Dependant Reconstruction..." << endl;
	CUT_SAFE_CALL( cutResetTimer(hTimer) );
    CUT_SAFE_CALL( cutStartTimer(hTimer) );
	//GPU Resolution Dependant Reconstruction
	//Thomas_GPU_Complete (nx, ny, nz, n_voxels, resolution, iterations, file, 0);
	CUT_SAFE_CALL(cutStopTimer(hTimer));
	timerValue = cutGetTimerValue(hTimer);
	cout << "GPU Resolution Dependant Reconstruction Finsihed." << endl;
	cout << "Reconstruction Time: " << timerValue / 1000 << "seg." << endl;
	////////////////////////////////////////////////////////////

	/////////////////GPU RII RECONSTRUCTION//////////////////
	cout << "Starting GPU Resolution Independant Reconstruction..." << endl;
	CUT_SAFE_CALL( cutResetTimer(hTimer) );
    CUT_SAFE_CALL( cutStartTimer(hTimer) );
	//CPU_Online_Reconstruction
	//Thomas_GPU_Complete (nx, ny, nz, n_voxels, resolution, iterations, file, 1);
	CUT_SAFE_CALL(cutStopTimer(hTimer));
	timerValue = cutGetTimerValue(hTimer);
	cout << "GPU Resolution Independant Finsihed." << endl;
	cout << "Reconstruction Time: " << timerValue / 1000 << "seg." << endl;
	///////////////////////////////////////////////////////////*/
 /*
}
void VerificarLecturaSino3D (void)
{
	FILE* fid;
	Sinogram3D* MiSino3D = new Sinogram3D("E:\\Sinogramas\\Sinograma GE\\sino3DNema.s");
	if((fid=fopen("E:\\Sinogramas\\Sinograma GE\\PruebaSino3d.txt","w"))==NULL)
	{
		printf("Error al abrir el archivo de salida");
	}
	for(int i = 0; i < MiSino3D->CantSegmentos; i++)
	{
		fprintf(fid, "Segmento: %d. Sinogramas: %d. MaxRingDiff: %d. MinRingDiff: %d.\n", i, MiSino3D->Segmentos[i]->CantSinogramas, MiSino3D->Segmentos[i]->MaxRingDiff, MiSino3D->Segmentos[i]->MinRingDiff);
		for(int j = 0; j < MiSino3D->Segmentos[i]->CantSinogramas; j++)
		{
			fprintf(fid, "\tSinograma: %d. LORS eje axial: %d.\n", j, MiSino3D->Segmentos[i]->Sinogramas2D[j]->CantZ);
			for(int k = 0; k < MiSino3D->Segmentos[i]->Sinogramas2D[j]->CantZ; k++)
			{
				fprintf(fid, "\t\tRing1: %d. Ring2: %d.\n", MiSino3D->Segmentos[i]->Sinogramas2D[j]->ListaRing1[k], MiSino3D->Segmentos[i]->Sinogramas2D[j]->ListaRing2[k]);
			}
		}
	}
	fclose(fid);
}

void TestConversionSino3DtoMicho(char* FullInputPath, char* OutputPath)
{
	char strNombreArchivo[100];
	SizeMichelogram MySizeMichelogram;
	Sinogram3D* MiSino3D = new Sinogram3D(FullInputPath);
	MiSino3D->FillConstant(1);
	sprintf(strNombreArchivo,"%s\\Sino3Dcte.dat", OutputPath);
	MiSino3D->SaveInFile(strNombreArchivo);
	MySizeMichelogram.NProj = MiSino3D->NProj;
	MySizeMichelogram.NR = MiSino3D->NR;
	MySizeMichelogram.NZ = MiSino3D->NRings;
	MySizeMichelogram.RFOV = MiSino3D->RFOV;
	MySizeMichelogram.ZFOV = MiSino3D->ZFOV;
	Michelogram* MiMichelogram =  new Michelogram(MySizeMichelogram);
	MiMichelogram->ReadDataFromSinogram3D(MiSino3D);
	sprintf(strNombreArchivo,"%s\\Micho_cte.dat", OutputPath);
	MiMichelogram->SaveInFile(strNombreArchivo);
}

void TestProjectionMicho(SizeVolume MySizeVolume, char* FullInputPath, char* OutputPath)
{
	char strNombreArchivo[100];
	SizeMichelogram MySizeMichelogram;
	Volume* MyVolume = new Volume(MySizeVolume);
	MyVolume->FillConstant(1.0);
	Sinogram3D* MiSino3D = new Sinogram3D(FullInputPath);
	MySizeMichelogram.NProj = MiSino3D->NProj;
	MySizeMichelogram.NR = MiSino3D->NR;
	MySizeMichelogram.NZ = MiSino3D->NRings;
	MySizeMichelogram.RFOV = MiSino3D->RFOV;
	MySizeMichelogram.ZFOV = MiSino3D->ZFOV;
	Michelogram* MiMichelogram =  new Michelogram(MySizeMichelogram);
	MiMichelogram->FillConstant(0);
	MLEM* PruebaMLEM = new MLEM(MiMichelogram, MyVolume, OutputPath,"ProjectionTest", 88.62/2);
	PruebaMLEM->ForwardProjection(MyVolume, MiMichelogram);
	sprintf(strNombreArchivo,"%s\\ProjectionTest.dat", OutputPath);
	MiMichelogram->SaveInFile(strNombreArchivo);
}


void ReconstruccionAATN (SizeVolume MySizeVolume, char* FullInputPath, char* OutputPath, char* Name, int CantIteraciones)
{
	SizeMichelogram MySizeMichelogram;
	Sinogram3D* MiSino3D = new Sinogram3D(FullInputPath);
	Volume* MyVolume = new Volume(MySizeVolume);
	MyVolume->FillConstant(1.0);
	MySizeMichelogram.NProj = MiSino3D->NProj;
	MySizeMichelogram.NR = MiSino3D->NR;
	MySizeMichelogram.NZ = MiSino3D->NRings;
	MySizeMichelogram.RFOV = MiSino3D->RFOV;
	MySizeMichelogram.ZFOV = MiSino3D->ZFOV;
	Michelogram* MiMichelogram =  new Michelogram(MySizeMichelogram);
	MiMichelogram->ReadDataFromSinogram3D(MiSino3D);
	MLEM* PruebaMLEM = new MLEM(MiMichelogram, MyVolume, OutputPath,Name, 88.62/2);
	PruebaMLEM->SensibilityFromFile = false;
	PruebaMLEM->GuardarLogTiempos = true;
	PruebaMLEM->CantIteraciones = CantIteraciones;
	PruebaMLEM->GuardarImagenIteracion = true;
	PruebaMLEM->Reconstruir();
}

void ReconstruccionAATN_GPU (SizeVolume MySizeVolume, char* FullInputPath, char* OutputPath, char* Name, int CantIteraciones)
{
	SizeMichelogram MySizeMichelogram;
	Sinogram3D* MiSino3D = new Sinogram3D(FullInputPath);
	Volume* MyVolume = new Volume(MySizeVolume);
	MyVolume->FillConstant(1.0);
	MySizeMichelogram.NProj = MiSino3D->NProj;
	MySizeMichelogram.NR = MiSino3D->NR;
	MySizeMichelogram.NZ = MiSino3D->NRings;
	MySizeMichelogram.RFOV = MiSino3D->RFOV;
	MySizeMichelogram.ZFOV = MiSino3D->ZFOV;
	Michelogram* MiMichelogram =  new Michelogram(MySizeMichelogram);
	MiMichelogram->ReadDataFromSinogram3D(MiSino3D);
	CUDA_MLEM* PruebaMLEM = new CUDA_MLEM(MiMichelogram, MyVolume, OutputPath,Name, 88.62/2);
	PruebaMLEM->SensibilityFromFile = false;
	PruebaMLEM->GuardarLogTiempos = true;
	PruebaMLEM->CantIteraciones = CantIteraciones;
	PruebaMLEM->GuardarImagenIteracion = true;
	PruebaMLEM->ReconstruirEnGPU();
}

/*

void ComparacionSiddonPattern (SizeMichelogram MySizeMichelogram, SizeVolume MySizeVolume)
{
	//////////// PARA CORER UNA SOLA VEZ ///////////////////
	char pathSiddonPattern[80];
	unsigned int hTimer;
	double timerValue;
	CUT_SAFE_CALL(cutCreateTimer(&hTimer));
	Michelogram* MyMichelogram = new Michelogram(MySizeMichelogram.NProj, MySizeMichelogram.NR, MySizeMichelogram.NZ, 1, 1, MySizeMichelogram.RFOV, MySizeMichelogram.ZFOV); // This scanner has 24 rings
	sprintf(pathSiddonPattern, "E:\\Reconstruction_Results\\SiddonPattern_%d_%d_%d.dat", MyMichelogram->NProj, MyMichelogram->NR, MyMichelogram->NZ);
	cout << "Starting Martin CPU Calculation of Siddon Pattern with Michelogram Class..." << endl;
	CUT_SAFE_CALL( cutResetTimer(hTimer) );
    CUT_SAFE_CALL( cutStartTimer(hTimer) );
	SaveSiddonPattern(MySizeMichelogram, MySizeVolume, pathSiddonPattern);
	CUT_SAFE_CALL(cutStopTimer(hTimer));
	timerValue = cutGetTimerValue(hTimer);
	cout << "Martin CPU Calculation of Siddon Pattern with Michelogram Class finsihed." << endl;
	cout << "Processing Time: " << timerValue / 1000 << "seg." << endl;

	sprintf(pathSiddonPattern, "E:\\Reconstruction_Results\\SiddonPattern_float_%d_%d_%d.dat", MyMichelogram->NProj, MyMichelogram->NR, MyMichelogram->NZ);
	cout << "Starting Martin CPU Calculation of Siddon Pattern with float..." << endl;
	CUT_SAFE_CALL( cutResetTimer(hTimer) );
    CUT_SAFE_CALL( cutStartTimer(hTimer) );
	SaveSiddonPattern_float(MySizeMichelogram, MySizeVolume, pathSiddonPattern);
	CUT_SAFE_CALL(cutStopTimer(hTimer));
	timerValue = cutGetTimerValue(hTimer);
	cout << "Martin CPU Calculation of Siddon Pattern with float finsihed." << endl;
	cout << "Processing Time: " << timerValue / 1000 << "seg." << endl;

	// GPU 
	MyCudaInitialization();

	sprintf(pathSiddonPattern, "E:\\Reconstruction_Results\\CUDA_SiddonPattern_%d_%d_%d.dat", MySizeMichelogram.NProj, MySizeMichelogram.NR, MySizeMichelogram.NZ);
	cout << "Starting Martin GPU Calculation of Siddon Pattern..." << endl;
	CUT_SAFE_CALL( cutResetTimer(hTimer) );
    CUT_SAFE_CALL( cutStartTimer(hTimer) );
	CUDA_SaveSiddonPattern(MySizeMichelogram, MySizeVolume, pathSiddonPattern);
	CUT_SAFE_CALL(cutStopTimer(hTimer));
	timerValue = cutGetTimerValue(hTimer);
	cout << "Martin GPU Calculation of Siddon Pattern finsihed." << endl;
	cout << "Processing Time: " << timerValue / 1000 << "seg." << endl;

}

void ComparacionSensibilityImage (SizeMichelogram MySizeMichelogram, SizeVolume MySizeVolume)
{
	unsigned int hTimer;
	char pathSensImage[80];
	double timerValue;
	CUT_SAFE_CALL(cutCreateTimer(&hTimer));
	Michelogram* MyMichelogram = new Michelogram(MySizeMichelogram.NProj, MySizeMichelogram.NR, MySizeMichelogram.NZ, 1, 1, MySizeMichelogram.RFOV, MySizeMichelogram.ZFOV); // This scanner has 24 rings
	
	cout << "Starting Martin CPU Calculation of Sensibility Image with Volume and Michelogram Class..." << endl;
	CUT_SAFE_CALL( cutResetTimer(hTimer) );
    CUT_SAFE_CALL( cutStartTimer(hTimer) );
	sprintf(pathSensImage, "E:\\Reconstruction_Results\\SensibilityVolume_%d_%d_%d.dat", MySizeVolume.Nx, MySizeVolume.Ny, MySizeVolume.Nz);
	float* SumAij = (float*) malloc(sizeof(float)*MySizeVolume.Nx*MySizeVolume.Ny*MySizeVolume.Nz);
	SumSystemMatrix(MyMichelogram, MySizeVolume, SumAij);
	CUT_SAFE_CALL(cutStopTimer(hTimer));
	timerValue = cutGetTimerValue(hTimer);
	SaveRawFile(SumAij, MySizeVolume.Nx*MySizeVolume.Ny*MySizeVolume.Nz, pathSensImage);
	cout << "Martin CPU Calculation of Sensibility Image with Volume and Michelogram Class finsihed." << endl;
	cout << "Processing Time: " << timerValue / 1000 << "seg." << endl;

	cout << "Starting Martin CPU Calculation of Sensibility Image with float..." << endl;
	CUT_SAFE_CALL( cutResetTimer(hTimer) );
    CUT_SAFE_CALL( cutStartTimer(hTimer) );
	sprintf(pathSensImage, "E:\\Reconstruction_Results\\SensibilityVolume_float_%d_%d_%d.dat", MySizeVolume.Nx, MySizeVolume.Ny, MySizeVolume.Nz);
	SumSystemMatrix_float(MySizeMichelogram, MySizeVolume, SumAij);
	CUT_SAFE_CALL(cutStopTimer(hTimer));
	timerValue = cutGetTimerValue(hTimer);
	SaveRawFile(SumAij, MySizeVolume.Nx*MySizeVolume.Ny*MySizeVolume.Nz, pathSensImage);
	cout << "Martin CPU Calculation of Sensibility Image with float finsihed." << endl;
	cout << "Processing Time: " << timerValue / 1000 << "seg." << endl;

	cout << "Starting Martin GPU Calculation of Sensibility Image..." << endl;
	CUT_SAFE_CALL( cutResetTimer(hTimer) );
    CUT_SAFE_CALL( cutStartTimer(hTimer) );
	sprintf(pathSensImage, "E:\\Reconstruction_Results\\CUDA_SensibilityVolume_%d_%d_%d.dat", MySizeVolume.Nx, MySizeVolume.Ny, MySizeVolume.Nz);
	CUDA_SaveSensibilityVolume(MySizeMichelogram, MySizeVolume, pathSensImage);
	CUT_SAFE_CALL(cutStopTimer(hTimer));
	timerValue = cutGetTimerValue(hTimer);
	cout << "Martin GPU Calculation of Sensibility Image finsihed." << endl;
	cout << "Processing Time: " << timerValue / 1000 << "seg." << endl;
}

void Comparacion_MLEM(Michelogram* MyMichelogram, SizeVolume MySizeVolume, unsigned int iterations, char* pathSensImage)
{
	unsigned int hTimer;
	double timerValue;
	CUT_SAFE_CALL(cutCreateTimer(&hTimer));
	Volume* MyVolume = new Volume(MySizeVolume.Nx, MySizeVolume.Ny, MySizeVolume.Nz);
	///////////////////////////////CPU_ONLINE MICHELOGRAM CLASS////////////////////////////////////////////////////
	cout << "Starting Martin CPU Online Reconstruction with Michelogram Class..." << endl;
	CUT_SAFE_CALL( cutResetTimer(hTimer) );
	CUT_SAFE_CALL( cutStartTimer(hTimer) );
	// Initialization of the Volume with a value different to zero
	MyVolume->FillConstant(0.2);
	MLEM(MyMichelogram, MyVolume, iterations, (char*) pathSensImage);
	CUT_SAFE_CALL(cutStopTimer(hTimer));
	timerValue = cutGetTimerValue(hTimer);
	cout << "Martin CPU Online Reconstruction with Michelogram Class finsihed." << endl;
	cout << "Reconstruction Time: " << timerValue / 1000 << "seg." << endl;
	//////////////////////////////////////////////////////////////////////////////////////////////////

		///////////////////////////////CPU_ONLINE float////////////////////////////////////////////////////
	cout << "Starting Martin CPU Online Reconstruction with float..." << endl;
	// Initialization of the Volume with a value different to zero
	MyVolume->FillConstant(0.2);
	float* float_volume = MyVolume->RawData();
	float* float_michelogram = MyMichelogram->RawData();
	SizeMichelogram MySizeMichelogram;
	MySizeMichelogram.NProj = MyMichelogram->NProj;
	MySizeMichelogram.NR = MyMichelogram->NR;
	MySizeMichelogram.NZ = MyMichelogram->NZ;
	MySizeMichelogram.RFOV = MyMichelogram->RFOV;
	MySizeMichelogram.ZFOV = MyMichelogram->ZFOV;
	CUT_SAFE_CALL( cutResetTimer(hTimer) );
	CUT_SAFE_CALL( cutStartTimer(hTimer) );
	MLEM(float_michelogram, float_volume, MySizeMichelogram, MySizeVolume, iterations, (char*) pathSensImage);
	CUT_SAFE_CALL(cutStopTimer(hTimer));
	timerValue = cutGetTimerValue(hTimer);
	cout << "Martin CPU Online Reconstruction with float finsihed." << endl;
	cout << "Reconstruction Time: " << timerValue / 1000 << "seg." << endl;
	/////////////////////////////////////////////////////////////////////////////////////////////////

	///////////////////////////////GPU_ONLINE////////////////////////////////////////////////////////
	// GPU initialization
 	MyCudaInitialization();
	MyVolume->FillConstant(0.2);
	//MLEM(MyMichelogram, MyVolume, iterations, "E:\\Final_Results\\Sensibility_Image_32_32_32.dat");
	cout << "Starting Martin GPU Online Reconstruction..." << endl;
	CUT_SAFE_CALL( cutResetTimer(hTimer) );
	CUT_SAFE_CALL( cutStartTimer(hTimer) );
	//CUDA_MLEM(MyMichelogram, MyVolume, iterations, (char*) pathSensImage,"E:\\Reconstruction_Results\\CUDA");
	CUDA_MLEM(MyMichelogram, MyVolume, iterations, "E:\\Reconstruction_Results\\CUDA2");
	CUT_SAFE_CALL(cutStopTimer(hTimer));
	timerValue = cutGetTimerValue(hTimer);
	cout << "Martin GPU Online Reconstruction finsihed." << endl;
	cout << "Reconstruction Time: " << timerValue / 1000 << "seg." << endl;
	//////////////////////////////////////////////////////////////////////////////////////////////////
}

void Test_GPU(Michelogram* MyMichelogram, SizeVolume MySizeVolume, unsigned int iterations, char* pathSensImage)
{
	unsigned int hTimer;
	double timerValue;
	char SavePath[80];
	CUT_SAFE_CALL(cutCreateTimer(&hTimer));
	Volume* MyVolume = new Volume(MySizeVolume.Nx, MySizeVolume.Ny, MySizeVolume.Nz);
	SizeMichelogram MySizeMichelogram;
	MySizeMichelogram.NProj = MyMichelogram->NProj;
	MySizeMichelogram.NR = MyMichelogram->NR;
	MySizeMichelogram.NZ = MyMichelogram->NZ;
	MySizeMichelogram.RFOV = MyMichelogram->RFOV;
	MySizeMichelogram.ZFOV = MyMichelogram->ZFOV;
	// GPU 
	MyCudaInitialization();

	sprintf(SavePath, "E:\\Reconstruction_Results\\CUDA_SiddonPattern_%d_%d_%d.dat", MySizeMichelogram.NProj, MySizeMichelogram.NR, MySizeMichelogram.NZ);
	cout << "Starting Martin GPU Calculation of Siddon Pattern..." << endl;
	CUT_SAFE_CALL( cutResetTimer(hTimer) );
    CUT_SAFE_CALL( cutStartTimer(hTimer) );
	CUDA_SaveSiddonPattern(MySizeMichelogram, MySizeVolume, SavePath);
	CUT_SAFE_CALL(cutStopTimer(hTimer));
	timerValue = cutGetTimerValue(hTimer);
	cout << "Martin GPU Calculation of Siddon Pattern finsihed." << endl;
	cout << "Processing Time: " << timerValue / 1000 << "seg." << endl;

	cout << "Starting Martin GPU Calculation of Sensibility Image..." << endl;
	CUT_SAFE_CALL( cutResetTimer(hTimer) );
    CUT_SAFE_CALL( cutStartTimer(hTimer) );
	sprintf(SavePath, "E:\\Reconstruction_Results\\CUDA_SensibilityVolume_%d_%d_%d.dat", MySizeVolume.Nx, MySizeVolume.Ny, MySizeVolume.Nz);
	CUDA_SaveSensibilityVolume(MySizeMichelogram, MySizeVolume, SavePath);
	CUT_SAFE_CALL(cutStopTimer(hTimer));
	timerValue = cutGetTimerValue(hTimer);
	cout << "Martin GPU Calculation of Sensibility Image finsihed." << endl;
	cout << "Processing Time: " << timerValue / 1000 << "seg." << endl;

	MyVolume->FillConstant(0.2);
	//MLEM(MyMichelogram, MyVolume, iterations, "E:\\Final_Results\\Sensibility_Image_32_32_32.dat");
	cout << "Starting Martin GPU Online Reconstruction..." << endl;
	CUT_SAFE_CALL( cutResetTimer(hTimer) );
	CUT_SAFE_CALL( cutStartTimer(hTimer) );
	CUDA_MLEM(MyMichelogram, MyVolume, iterations, (char*) pathSensImage,"E:\\Reconstruction_Results\\CUDA");
	//CUDA_MLEM(MyMichelogram, MyVolume, iterations, "E:\\Reconstruction_Results\\CUDA2");
	CUT_SAFE_CALL(cutStopTimer(hTimer));
	timerValue = cutGetTimerValue(hTimer);
	cout << "Martin GPU Online Reconstruction finsihed." << endl;
	cout << "Reconstruction Time: " << timerValue / 1000 << "seg." << endl;
}

void Testbench2D ()
{
	// Voy leyendo los archivos con sinogramas 2D
	char fullPath[80];
	char fullPathSalida[120];
	Sinogram2D* MySinogram = new Sinogram2D(192,192,30,50);
	// Guardo el sensibility volume
	ImageSize MyImageSize;
	MyImageSize.Nx = 128;
	MyImageSize.Ny = 128;
	MyImageSize.RFOV = 30;
	MyImageSize.Rscanner = 50;
	float* SumAij = (float*) malloc(sizeof(float)* MyImageSize.Nx * MyImageSize.Ny);
	SumSystemMatrix(MySinogram, MyImageSize, SumAij);
	sprintf(fullPathSalida, "E:\\Reconstruction_Results\\Testbench2D\\SensibilityImage_128x128.dat");
	SaveRawFile(SumAij, MyImageSize.Nx * MyImageSize.Ny, fullPathSalida);
	// El testbench est� formado por 9 sinogramas de 192x192 con cantidad de eventos variable.
	Image* MyImage = new Image(128,128);
	MyImage->FillConstant(0.2);
	unsigned int iteraciones = 10;
	for(float i=4; i<=8; i+=0.5)
	{
		sprintf(fullPath, "E:\\Sinogramas\\Sinogramas2D\\SheppLogan_192x192_10^%f.dat", i);
		MySinogram->ReadFromFile(fullPath);
		sprintf(fullPathSalida, "E:\\Sinogramas\\Sinogramas2D\\Sino.dat");
		MySinogram->SaveInFile(fullPathSalida);
		MLEM(MySinogram, MyImage, iteraciones);
		sprintf(fullPathSalida, "E:\\Reconstruction_Results\\Testbench2D\\SehppLogan_128x128_10^%f.dat", i);
		MyImage->SaveInFile(fullPathSalida);
	}
}

*/