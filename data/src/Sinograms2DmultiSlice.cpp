/**
	\file Sinograms2DmultiSlice.cpp
	\brief Archivo que contiene la implementación de la clase Sinograms2DmultiSlice.

	Este archivo define la clase Sinograms2DmultiSlice. 
	\todo 
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2012.10.01
	\version 1.0.0
*/

#include <Sinograms2DmultiSlice.h>


//using namespace::std;
//using namespace::iostream;

Sinograms2DmultiSlice::Sinograms2DmultiSlice(int nProj, int nR, float rFov_mm, float zFov_mm, int nSinograms) : Sinogram2D(numProj, numR, radioFov_mm)
{
  numSinograms = nSinograms;
  axialFov_mm = zFov_mm;
  initParameters();
}

/// Constructor de copia
Sinograms2DmultiSlice::Sinograms2DmultiSlice(Sinograms2DmultiSlice* srcSegment)
{
  this->numSinograms = srcSegment->numSinograms;
  numProj = srcSegment->getNumProj();
  numR = srcSegment->getNumR();
  radioFov_mm = srcSegment->getRadioFov_mm();
  axialFov_mm = srcSegment->getAxialFoV_mm();
  initParameters();
}

/// Constructor desde un archivo:
Sinograms2DmultiSlice::Sinograms2DmultiSlice(string fileName, float rF_mm, float zF_mm)
{
  this->radioFov_mm = rF_mm;
  this->axialFov_mm = zF_mm;
}

/// Destructor de la clase Segmento.
Sinograms2DmultiSlice::~Sinograms2DmultiSlice()
{
}

void Sinograms2DmultiSlice::CopyBins(Sinograms2DmultiSlice* source)
{
  for(int i = 0; i< source->getNumSinograms(); i++)
  {
    for(int j = 0; j < source->getNumProj(); j++)
    {
      for(int k = 0; k < source->getNumR(); k++)
      {
	this->getSinogram2D(i)->setSinogramBin(j, k, source->getSinogram2D(i)->getSinogramBin(j,k));
      }
    }
  }
}
void Sinograms2DmultiSlice::initParameters()
{
  float zIncrement;
  // Initialization of Z values
  ptrAxialValues_mm = (float*) malloc(numSinograms*sizeof(float));
  zIncrement = (float)axialFov_mm/numSinograms;
  for(int i = 0; i < numSinograms; i ++)
  {
	  // Initialization of Z Values
	  ptrAxialValues_mm[i] = zIncrement/2 + i * zIncrement;
  }
}

/// Método que lee los sinogramas desde un archivo de interfile.
bool Sinograms2DmultiSlice::readFromInterfile(string headerFilename)
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
  if(fi->dim[0] >= 3)
  {
	  numR = fi->image[0].width;
	  numProj = fi->image[0].height;
	  
	  // En principio al tamaño de pixel no le doy bola
	  
	  if(fi->dim[0] == 3)
	  {
		// Tengo múltiples sinogramas 2D.
		numSinograms = fi->number;
	  }
	  else if(fi->dim[0] > 3)
	  {
		  this->strError = "No se soportan sinogramas de más de 3 dimensiones en esta clase";
		  return false;
	  }

  }
  else
  {
		  this->strError = "No se admiten sinogramas de menos de 3 dimensiones";
		  return false;
  }

  // Ahora cargo los píxeles de la imagen. Esto depende del tipo de datos de la misma.
  // Por ahora solo soporto floats, en el futuro se convertirá a través de un template
  // en una clase que soporte todos los tiposde datos.
  // Tipos de datos en medcon: BIT1, BIT8_U, BIT16_S, BIT32_S, FLT32, FLT64, COLRGB
  if(fi->type == FLT32)
  {
    // Ahora pido memoria para todos los sinogramas:
    initSinograms();
    int nBinsSino2d = numProj * numR;
    for(int i = 0; i < numSinograms; i++)
    {
	// La memoria supuestamente la pide en el constructor.
	IMG_DATA *id;
	id = &fi->image[i];
	float* ptrSinogram = this->getSinogram2D(i)->getSinogramPtr();
	// We accept empty sinograms as a sample, in that case fill it with zeros, if not copy from the binary:
	if (fi->truncated == MDC_YES)
	  memset(ptrSinogram, 0, sizeof(float)*nBinsSino2d);
	else
	  memcpy(ptrSinogram, id->buf, sizeof(float)*nBinsSino2d);
	//this->getSinogram2D(i)->initParameters(); it should be intialized in initSinograms()
    }
	
  }
  else
  {
    this->strError = "Only float images are accepted.";
    return false;
  }
  return true;
}

bool Sinograms2DmultiSlice::writeInterfile(string headerFilename)
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
  fileStream << "!total number of images := " << numSinograms << eol;
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
  fileStream << "!matrix size [1] := " << this->getSinogram2D(0)->getNumR() << eol;
  // Cantidad de elementos en y (filas) o de ángulos:
  fileStream << "!matrix size [2] := " << this->getSinogram2D(0)->getNumProj() << eol;
  // Por ahora el único tipo de dato del sinograma es float:
  fileStream << "!number format := short float" << eol;
  fileStream << "!number of bytes per pixel := " << sizeof(float) << eol;
  /* Por ahora no lo pongo al scaling factor, porque si no coincide con el de generación de datos me caga.
  fileStream << "scaling factor (mm/pixel) [1] := " << (this->getRValue(1)-this->getRValue(0)) << eol;
  fileStream << "scaling factor (deg/pixel) [2] := " << (this->getAngValue(1)-this->getAngValue(0)) << eol;
  */
  fileStream << "!number of projections := " << numSinograms << eol;
  fileStream << "!extent of rotation := " << (this->getSinogram2D(0)->getAngValue(this->getSinogram2D(0)->getNumProj()-1) - this->getSinogram2D(0)->getAngValue(0)) << eol;
  // Máximo valor de píxels:
  float max = this->getSinogram2D(0)->getSinogramBin(0,0);
  for(int k = 0; k < this->numSinograms; k++)
  {
    for(int i = 0; i < this->getSinogram2D(k)->getNumProj(); i++)
    {
      for(int j = 0; j < this->getSinogram2D(k)->getNumR(); j++)
      {
	if(this->getSinogram2D(k)->getSinogramBin(i,j) > max)
	      max = this->getSinogram2D(k)->getSinogramBin(i,j);
      }
    }
  }
  fileStream << "!maximum pixel count := " << max << eol;
  fileStream << "!END OF INTERFILE :=\n" << eol;
  fileStream.close();
  
  // Ya terminé con el header, ahora escribo el archivo binario:
  fileStream.open(dataFilename.c_str(), ios_base::binary);
  for(int i = 0; i < numSinograms; i++)
  {
      fileStream.write((char*)this->getSinogram2D(i)->getSinogramPtr(), this->getSinogram2D(i)->getNumProj()*this->getSinogram2D(i)->getNumR()*sizeof(float)); 
  }
  fileStream.close();
  return true;
}

bool Sinograms2DmultiSlice::FillConstant(float Value)
{
  for(int i = 0; i < this->getNumSinograms(); i++)
  {
    for(int j = 0; j < this->getSinogram2D(i)->getNumProj(); j++)
    {
      for(int k = 0; k < this->getSinogram2D(i)->getNumR(); k++)
      {
		this->getSinogram2D(i)->setSinogramBin(j,k, Value);
      }
    }
  }
  return true;
}
