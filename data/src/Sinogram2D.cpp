/**
	\file Sinogram2D.cpp
	\brief Archivo que contiene la implementación de la clase Sinogram2D.

	Este archivo define la clase Sinogram2. La misma define un sinograma de dos dimensiones genérico
	pensando en PET, o sea que considera que el ángulo de las proyecciones va entre 0 y 180º.
	\todo Extenderlo de manera genérico a distintas geometrías.
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2010.10.01
	\version 1.0.0
*/

#include <math.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <Utilities.h>
#include <Geometry.h>
#include <Sinogram2D.h>
#include <iostream>
#include <fstream>
#include <medcon.h>
#include <m-intf.h>

Sinogram2D::Sinogram2D()
{
  /* Para un sinograma genérico los ángulos de proyecciones van de 0 a 180º. */
  minAng_deg = 0;
  maxAng_deg = 180;
  // Inicializo los punteros con 1 byte, después se realoca.
  ptrSinogram = (float*) malloc(1*sizeof(float));
  ptrAngValues_deg = (float*) malloc(1*sizeof(float));
  ptrRvalues_mm = (float*) malloc(1*sizeof(float));
}

Sinogram2D::Sinogram2D(float myRfov_mm)
{
  /* Para un sinograma genérico los ángulos de proyecciones van de 0 a 180º. */
  minAng_deg = 0;
  maxAng_deg = 180;
  // Inicializo los punteros con 1 byte, después se realoca.
  ptrSinogram = (float*) malloc(1*sizeof(float));
  ptrAngValues_deg = (float*) malloc(1*sizeof(float));
  ptrRvalues_mm = (float*) malloc(1*sizeof(float));
  // rfov:
  radioFov_mm = myRfov_mm;
}

Sinogram2D::Sinogram2D(unsigned int myNumProj, unsigned int myNumR, float myRadioFov)
{
  /* Para un sinograma genérico los ángulos de proyecciones van de 0 a 180º. */
  minAng_deg = 0;
  maxAng_deg = 180;
  radioFov_mm = myRadioFov;
  numR = myNumR;
  numProj = myNumProj;
  // Allocates Memory for th Sinogram
  ptrSinogram = (float*) malloc(numProj*numR*sizeof(float));
  // Allocates Memory for the value's vectors
  ptrAngValues_deg = (float*) malloc(numProj*sizeof(float));
  ptrRvalues_mm = (float*) malloc(numR*sizeof(float));
  // Initialization
  float RIncrement = (2 * radioFov_mm) / (numR);
  float PhiIncrement = (float)maxAng_deg / (numProj);
  for(unsigned int i = 0; i < numProj; i ++)
  {
	  // Initialization of Phi Values
	  ptrAngValues_deg[i] = PhiIncrement/2 + i * PhiIncrement;
	  for(unsigned int j = 0; j < numR; j++)
	  {
		  if(i == 0)
		  {
			  // ptrRvalues initialization is necesary just one time
			  
			  ptrRvalues_mm[j] = RIncrement/2 + j * RIncrement - radioFov_mm;
		  }
		  ptrSinogram[i * numR + j] = 0;
	  }
  }
}

Sinogram2D::~Sinogram2D()
{
  /// Limpio la memoria
  free(ptrSinogram);
  free(ptrRvalues_mm);
  free(ptrAngValues_deg);
}

/// Constructor de Copia
Sinogram2D::Sinogram2D(const Sinogram2D* srcSinogram2D)
{
  radioFov_mm = srcSinogram2D->radioFov_mm;
  numR = srcSinogram2D->numR;
  numProj = srcSinogram2D->numProj;
  // Allocates Memory for th Sinogram and copy data
  ptrSinogram = (float*) malloc(numProj*numR*sizeof(float));
  memcpy(ptrSinogram, srcSinogram2D->ptrSinogram, numProj*numR*sizeof(float));
  // Allocates cand copy Memory for the value's vectors
  ptrAngValues_deg = (float*) malloc(numProj*sizeof(float));
  memcpy(ptrAngValues_deg, srcSinogram2D->ptrAngValues_deg, numProj*sizeof(float));
  ptrRvalues_mm = (float*) malloc(numR*sizeof(float));
  memcpy(ptrRvalues_mm, srcSinogram2D->ptrRvalues_mm, numR*sizeof(float));
}

// Constructor que genera un sinograma reducido en proyecciones para usar en osem
Sinogram2D::Sinogram2D(const Sinogram2D* srcSinogram2D, int indexSubset, int numSubsets)
{

  // Calculo cuantas proyecciones va a tener el subset:
  int numProjSubset = floor((float)srcSinogram2D->numProj / (float)numSubsets);
  // Siempre calculo por defecto, luego si no dio exacta la división, debo agregar un ángulo a la proyección:
  if((srcSinogram2D->numProj%numSubsets)>indexSubset)
    numProjSubset++;
  // Ahora copio todos los parámetros del sinograma del que quiero obtener el subser
  // y solo cambio la cantidad de proyecciones y genero los ángulos de las proyecciones
  // según
  minAng_deg = srcSinogram2D->minAng_deg;
  maxAng_deg = srcSinogram2D->maxAng_deg;
  radioFov_mm = srcSinogram2D->radioFov_mm;
  numR = srcSinogram2D->numR;
  numProj = numProjSubset;
  // Allocates Memory for th Sinogram
  ptrSinogram = (float*) malloc(numProj*numR*sizeof(float));
  // Allocates Memory for the value's vectors
  ptrAngValues_deg = (float*) malloc(numProj*sizeof(float));
  ptrRvalues_mm = (float*) malloc(numR*sizeof(float));
  // Initialization
  // Los valores de r y de los ángulos de las proyecciones, los obtengo del sinograma original. Para
  // el caso de los ángulos solo debo tomar uno cada numSubsets. Ya que hago esto, también copio los valores
  // de los sinogramas:
  for(unsigned int i = 0; i < numProj; i ++)
  {
    // indice angulo del sino completo:
    int iAngCompleto = indexSubset + numSubsets*i;
    // Initialization of Phi Values
    ptrAngValues_deg[i] = srcSinogram2D->ptrAngValues_deg[iAngCompleto];
    for(unsigned int j = 0; j < numR; j++)
    {
      if(i == 0)
      {
	// ptrRvalues initialization is necesary just one time
	
	ptrRvalues_mm[j] = srcSinogram2D->ptrRvalues_mm[j];
      }
      ptrSinogram[i * numR + j] = srcSinogram2D->ptrSinogram[iAngCompleto * numR + j];
    }
  }
}

void Sinogram2D::initParameters()
{
  // Allocates Memory for the value's vectors
  ptrAngValues_deg = (float*) malloc(numProj*sizeof(float));
  ptrRvalues_mm = (float*) malloc(numR*sizeof(float));
  // Initialization
  float RIncrement = (2 * radioFov_mm) / numR;
  float PhiIncrement = (float)maxAng_deg / numProj;
  for(unsigned int i = 0; i < numProj; i ++)
  {
	  // Initialization of Phi Values
	  ptrAngValues_deg[i] = PhiIncrement/2 + i * PhiIncrement;
  }
  for(unsigned int j = 0; j < numR; j++)
  {	  
	ptrRvalues_mm[j] = RIncrement/2 + j * RIncrement - radioFov_mm;
  }
}

void Sinogram2D::setRadioFov_mm(float rFov_mm)
{ 
  radioFov_mm = rFov_mm;
  float rIncrement = (2 * radioFov_mm) / numR;
  for(unsigned int j = 0; j < numR; j++)
  {	  
	ptrRvalues_mm[j] = rIncrement/2 + j * rIncrement - radioFov_mm;
  }
}

void Sinogram2D::divideBinToBin(Sinogram2D* sinogramDivisor)
{
  float numerador, denominador;
  for(int i = 0; i < numProj; i ++)
  {
	for(int j = 0; j < numR; j++)
	{
	  // Para 0/0 lo dejo en cero. Como este cociente se hace para la reconstrucción, en realidad
	  // si el numerador es cero, directamente lo dejo en cero, porque no tengo cuentas en ese bin.
	  numerador = this->getSinogramBin(i,j);
	  denominador = sinogramDivisor->getSinogramBin(i,j);
	  if((numerador != 0)&&(denominador!=0))
	  {
		this->setSinogramBin(i,j, numerador/denominador);
	  }
	  else
	  {
		this->setSinogramBin(i,j, 0);
	  }
	}
  }
}

void Sinogram2D::multiplyBinToBin(Sinogram2D* sinogramFactor)
{
  for(int i = 0; i < numProj; i ++)
  {
	for(int j = 0; j < numR; j++)
	{
	  this->setSinogramBin(i,j, this->getSinogramBin(i,j)*sinogramFactor->getSinogramBin(i,j));
	}
  }
}

void Sinogram2D::inverseDivideBinToBin(Sinogram2D* sinogramDividend)
{
  float numerador, denominador;
  for(int i = 0; i < numProj; i ++)
  {
	for(int j = 0; j < numR; j++)
	{
	  // Para 0/0 lo dejo en cero. Como este cociente se hace para la reconstrucción, en realidad
	  // si el numerador es cero, directamente lo dejo en cero, porque no tengo cuentas en ese bin.
	  denominador = this->getSinogramBin(i,j);
	  numerador = sinogramDividend->getSinogramBin(i,j);
	  if((numerador != 0)&&(denominador!=0))
	  {
		this->setSinogramBin(i,j, numerador/denominador);
	  }
	  else
	  {
		this->setSinogramBin(i,j, 0);
	  }
	}
  }
}


bool Sinogram2D::Fill(Event2D* Events, unsigned int NEvents)
{
  for(unsigned int i = 0; i < NEvents; i++)
  {
	  int BinPhi, BinR;
	  float m = (Events[i].Y2 - Events[i].Y1)/(Events[i].X2 - Events[i].X1);
	  float Phi = atan((Events[i].Y2 - Events[i].Y1)/(Events[i].X2 - Events[i].X1)) * RAD_TO_DEG + PI_OVER_2;
	  // For a LOR defined as y = mx + b, with m = tan(-1/Phi) -> R = b*cos(-1/Phi)
	  float b = Events[i].Y1 - (Events[i].X1 * m);
	  float R = b * cos(-1/Phi);
	  // Now I already have (Phi,R) coordinates fo the LOR. I need to
	  // find in which bin of the sinogram this LOR fits.
	  if(((BinPhi = SearchBin(ptrAngValues_deg,numProj,Phi))!=-1)&&((BinR = SearchBin(ptrRvalues_mm,numR,R))!=-1))
	  {
		  ptrSinogram[BinPhi*numR+BinR]++;
	  }
  }
  return true;
}

bool Sinogram2D::FillConstant(float Value)
{
  for(unsigned int i = 0; i < numProj; i ++)
  {
	  for(unsigned int j = 0; j < numR; j++)
	  {
		  /// Copio el dato desde el array pasado, al array de la clase que contiene los 
		  /// datos del sinograma.
		  ptrSinogram[i * numR + j] = Value;
	  }
  }
  return true;
}

bool Sinogram2D::SaveInFile(char* filePath)
{
  FILE* fileSinogram = fopen(filePath,"wb");
  unsigned int CantBytes;
  const unsigned int SizeData = numProj * numR;
  if (fileSinogram != NULL)
  {
	  if((CantBytes =  fwrite(ptrSinogram, sizeof(float), numProj*numR , fileSinogram)) !=  (numProj*numR))
		  return false;
  }
  else
	  return false;
  fclose(fileSinogram);
  return true;
}

bool Sinogram2D::readFromInterfile(string headerFilename)
{
  const char* msg = NULL;
  // Uso las funciones de la aplicaciónde libre distribución Medcon.
  // Ver después de embeber los métodos en una clase propia.
  FILEINFO* fi;	// Puntero a estructura del tipo FILEINFO definida en Medcon.
  int error;
  // Pido memoria para la estructura:
  fi = (FILEINFO*) malloc(sizeof(FILEINFO));
  // Abro el archivo en inicializo fi.
  if ((error = MdcOpenFile(fi, headerFilename.c_str())) != MDC_OK) return false;
  // Verifico que el formato sea intefile, sino salgo con false.
  // Leo el formato del archivo
  if ( MdcGetFrmt(fi) != MDC_FRMT_INTF ) {
	MdcCloseFile(fi->ifp);
	this->strError.assign("El archivo ");
	this->strError += fi->ifname;
	this->strError += " no es compatible con el formato interfile";
	//this->strError.assign("El archivo " +"de");// fi->ifname + " no es compatible con el formato interfile");
	return false;
  }
  /// Para que me reconozca la tercera dimensión.
  #define MDC_INTF_SUPPORT_DIALECT 1
  // Leo el archivo intefile.
  if((msg=MdcReadINTF(fi)) != NULL)
  {
	  // Hubo un error en la lectura.
	  this->strError = msg;
	  return false;
  }
  // Estoy en condiciones de cargar los datos en los campos del objeto.
  // ver m-structs.h para ver detalle de todos los campos de fi.
  // Primero guardo los tamaños (solo acepto hasta 3 dimensiones). Tengo en cuenta
  // que la primera dimensión (x), o sea el ancho, representa las distancias R, y 
  // lasegunda dimensión (y) los ángulos.
  if(fi->dim[0] >= 2)
  {
	  this->numR = fi->image[0].width;
	  this->numProj = fi->image[0].height;
	  // En principio al tamaño de pixel no le doy bola
	  
	  if(fi->dim[0] == 3)
	  {
		// Este es un sinograma 2D, por lo que solo debería haber una imagen.
		if(fi->number>1)
		{
		  // Hay más de una imagen, no sirve para el sinograma 2d del tgs:
		  this->strError = "El sinograma tiene múltiples imágenes 2d cuando debería ser una sola (Sinograma2Dtgs).";
		  return false;
		}
	  }
	  else if(fi->dim[0] > 3)
	  {
		  this->strError = "No se soportan sinogramas de más de 3 dimensiones";
		  return false;
	  }

  }
  else
  {
		  this->strError = "No existen sinogramas de menos de 2 dimensiones";
		  return false;
  }

  // Ahora cargo los píxeles de la imagen. Esto depende del tipo de datos de la misma.
  // Por ahora solo soporto floats, en el futuro se convertirá a través de un template
  // en una clase que soporte todos los tiposde datos.
  // Tipos de datos en medcon: BIT1, BIT8_U, BIT16_S, BIT32_S, FLT32, FLT64, COLRGB
  if(fi->type == FLT32)
  {
	// Pedir memoria. Lo hago con realloc porque en el constructor ya se pide memoria, pero
	// por otro lado debo pedirla acá si o si porque en el archivo intefile me pueden
	// cambiar als dimensiones que se apsaron al constructor.
	this->ptrSinogram = (float*) realloc(this->ptrSinogram, sizeof(float)*this->numProj*this->numR);
	int nBinsSino2d = this->numProj * this->numR;
	IMG_DATA *id;
	for(int i = 0; i < fi->number; i++)
	{
	  id = &fi->image[i];
	  memcpy(this->ptrSinogram + i*nBinsSino2d, id->buf, sizeof(float)*nBinsSino2d);
	}
  }
  else
  {
	  this->strError = "Al momento solo se soportan sinogramas del tipo float.";
	  return false;
  }
  initParameters();
  return true;
}

bool Sinogram2D::writeInterfile(string headerFilename)
{
  // Objeto ofstream para la escritura en el archivo de log.
  ofstream fileStream;
  string dataFilename;
  string eol;
  // El nombre del archivo puede incluir un path.
  dataFilename.assign(headerFilename);
  headerFilename.append(".h33");
  dataFilename.append(".i33");
  eol.assign("\r\n");
  // Abro y creo el archivo:
  fileStream.open(headerFilename.c_str(), ios_base::out);
  
  // Empiezo a escribir el sinograma en formato interfile:
  fileStream << "!INTERFILE :=" << eol;
  fileStream << "!imaging modality := nucmed" << eol;
  fileStream << "!originating system := ar-pet" << eol;
  fileStream << "!version of keys := 3.3" << eol;
  fileStream << "!date of keys := 1992:01:01" << eol;
  fileStream << "!conversion program := ar-pet" << eol;
  fileStream << "!program author := ar-pet" << eol;
  fileStream << "!program version := 1.10" << eol;
  // Necesito guardar fecha y hora
  time_t rawtime;
  struct tm * timeinfo;
  time ( &rawtime );
  timeinfo = localtime ( &rawtime );
  fileStream << "!program date := " << asctime(timeinfo) << eol;
  fileStream << "!GENERAL DATA := " << eol;
  fileStream << "original institution := cnea" << eol;
  fileStream << "contact person := m. belzunce" << eol;
  fileStream << "data description := tomo" << eol;
  fileStream << "!data starting block := 0" << eol;

  // Para el nombre del archivo de datos, debe ser sin el subdirectorio, si el filename se
  // diera con un directorio. Porque tanto el h33 como el i33 se guardarán en ese directorio.
  // Por eso chequeo si es un nombre con path cinluido, o sin el, y en caso de serlo me quedo solo
  // con el archivo.
  size_t lastDash = dataFilename.find_last_of("/\\");
  fileStream << "!name of data file := " << dataFilename.substr(lastDash+1) << eol;
  fileStream << "patient name := Phantom" << eol;
  fileStream << "!patient ID  := 12345" << eol;
  fileStream << "patient dob := 1968:08:21" << eol;
  fileStream << "patient sex := M" << eol;
  fileStream << "!study ID := simulation" << eol;
  fileStream << "exam type := simulation" << eol;
  fileStream << "data compression := none" << eol;
  fileStream << "data encode := none" << eol;
  fileStream << "data compression := none" << eol;
  fileStream << "data compression := none" << eol;
  fileStream << "data compression := none" << eol;

  // Datos de la proyección (sinograma 2d):
  fileStream << "!GENERAL IMAGE DATA :=" << eol;
  fileStream << "!type of data := Tomographic" << eol;
  // Una sola imagen.
  fileStream << "!total number of images := " << 1 << eol;
  fileStream << "!imagedata byte order := LITTLEENDIAN" << eol;
  fileStream << "!number of energy windows := 1" << eol;
  fileStream << "energy window [1] := F18m" << eol;
  fileStream << "energy window lower level [1] := 430" << eol;
  fileStream << "energy window upper level [1] := 620" << eol;
  fileStream << "flood corrected := N" << eol;
  fileStream << "decay corrected := N" << eol;

  // Hay varios datos, que por ahora no interesan:
  fileStream << "!SPECT STUDY (general) :=" << eol;
  fileStream << "!number of detector heads := 1" << eol;
  fileStream << "!number of images/window := 1" << eol;
  // Cantidad de elementos en x (columnas) o se de posiciones R:
  fileStream << "!matrix size [1] := " << this->getNumR() << eol;
  // Cantidad de elementos en y (filas) o de ángulos:
  fileStream << "!matrix size [2] := " << this->getNumProj() << eol;
  // Por ahora el único tipo de dato del sinograma es float:
  fileStream << "!number format := short float" << eol;
  fileStream << "!number of bytes per pixel := " << sizeof(float) << eol;
  /* Por ahora no lo pongo al scaling factor, porque si no coincide con el de generación de datos me caga.
  fileStream << "scaling factor (mm/pixel) [1] := " << (this->getRValue(1)-this->getRValue(0)) << eol;
  fileStream << "scaling factor (deg/pixel) [2] := " << (this->getAngValue(1)-this->getAngValue(0)) << eol;
  */
  fileStream << "!extent of rotation := " << (this->maxAng_deg - this->minAng_deg) << eol;
  // Máximo valor de píxels:
  float max = this->getSinogramBin(0,0);
  for(int i = 0; i < this->numProj; i++)
  {
	for(int j = 0; j < this->numR; j++)
	{
	  if(this->getSinogramBin(i,j) > max)
		max = this->getSinogramBin(i,j);
	}
  }
  fileStream << "!maximum pixel count := " << max << eol;
  fileStream << "!END OF INTERFILE :=\n" << eol;
  fileStream.close();
  
  // Ya terminé con el header, ahora escribo el archivo binario:
  fileStream.open(dataFilename.c_str(), ios_base::binary);
  fileStream.write((char*)this->ptrSinogram, this->numProj*this->numR*sizeof(float));
  fileStream.close();
}

// Method that reads the Sinogram data from a file. The dimensions of the
// expected Sinogram are the ones loaded in the constructor of the class
bool Sinogram2D::readFromFile(string filePath)
{
  FILE* fileSinogram = fopen(filePath.c_str(),"rb");
  unsigned int CantBytes;
  const unsigned int SizeData = numProj * numR;
  // Cargo los daots en el punetro Sinogram
  if (fileSinogram != NULL)
  {
	  if((CantBytes =  fread(ptrSinogram,sizeof(float), SizeData , fileSinogram)) != SizeData)
		  return false;
  }
  else
	  return false;
  fclose(fileSinogram);
  return true;
}

// Method that reads the Sinogram data from a float array. The dimensions of the
// expected Sinogram are the ones loaded in the constructor of the class
bool Sinogram2D::ReadFromArray(float* SinogramArray)
{
  for(unsigned int i = 0; i < numProj; i ++)
  {
	  for(unsigned int j = 0; j < numR; j++)
	  {
		  /// Copio el dato desde el array pasado, al array de la clase que contiene los 
		  /// datos del sinograma.
		  ptrSinogram[i * numR + j] = SinogramArray[i * numR + j];
	  }
  }
  return true;
}

float* Sinogram2D::getAnglesInRadians()
{
  float* angles_radians;
  angles_radians = (float*) malloc(numProj*sizeof(float));
  for(unsigned int i = 0; i < numProj; i ++)
  {
    angles_radians[i] = ptrAngValues_deg[i] * DEG_TO_RAD;
  }
  return angles_radians;
}

float Sinogram2D::getLikelihoodValue(Sinogram2D* referenceProjection)
{
  float Likelihood = 0;
  for(unsigned int k = 0; k < this->getNumProj(); k++)
  {
	for(unsigned int l = 0; l < this->getNumR(); l++)
	{
	  if(this->getSinogramBin(k, l) >0)
	  {
		Likelihood += referenceProjection->getSinogramBin(k,l) 
			* log(this->getSinogramBin(k,l)) 
			- this->getSinogramBin(k,l) ;
	  }
	}
  }
  return Likelihood;
}

bool Sinogram2D::getFovLimits(int indexAng, int indexR, Point2D* limitPoint1, Point2D* limitPoint2)
{
  // Para el caso de un sinograma2d genérico, las coordenadas de las lors, son las mismas del límite del
  // fov porque no tengo información del scanner:
  this->getPointsFromLor(indexAng, indexR, limitPoint1, limitPoint2);
}

