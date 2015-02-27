/**
	\file Images.cpp
	\brief Archivo con los métodos de la clase Image.
	
	\author Martín Belzunce (martin.sure@gmail.com)
	\date 2010.09.03
	\version 1.0.0
	
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <Utilities.h>
#include <string>
#include <string.h>
#include <Images.h>
#include "medcon.h"
#include "m-intf.h"

Image::Image()
{
	/// Para el realloc del read from interfile.
	this->pixels = NULL;
}

Image::Image(string imageFilename)
{
	/// Para el realloc del read from interfile.
	this->pixels = NULL;
	this->readFromInterfile((char*)imageFilename.c_str());
}

Image::Image(SizeImage sizeVolume)
{
	// The image is kept in a array format which is saved going from
	// left to right, and then up to down. First increases column, and then
	// row.
	this->size = sizeVolume;
	if (this->size.nDimensions == 2)
	{
		/// Si la dimensión es 2 la cantidad de píxeles en Z es 1, porque es como si tuviera un solo plano.
		this->size.nPixelsZ = 1;
	}
	this->nPixels = this->getPixelCount();
	// Asigno la memoria para los píxeles.
	pixels = new float[this->nPixels];
	// Inicializo los píxeles en cero.
	for(int i = 0; i < this->nPixels; i++)
	{
		this->pixels[i] = 0;
	}
	// Calculo los tama�os del fov:
	this->fovHeight_mm = this->size.sizePixelZ_mm * this->size.nPixelsZ;
	this->fovRadio_mm = (this->size.sizePixelX_mm * this->size.nPixelsX)/2;
}

/// Constructor que genera una copia de una imagen.
Image::Image(Image* copyImage)
{
  this->size = copyImage->getSize();
  // Copio el resto de las propiedades:
  this->fovRadio_mm = copyImage->fovRadio_mm;
  this->fovHeight_mm = copyImage->fovHeight_mm;
  this->nPixels = copyImage->nPixels;
  
  // Asigno la memoria para los píxeles.
  pixels = new float[this->nPixels];
  // Copio el contenido de la imagen.
  for(int i = 0; i < copyImage->nPixels; i++)
  {
	  this->pixels[i] = copyImage->pixels[i];
  }
  
  return;
}

Image::~Image()
{
	free(pixels);
}

/// Constructor que genera una copia de una imagen.
bool Image::CopyFromImage(Image* copyImage)
{
  if((this->size.nPixelsX != copyImage->getSize().nPixelsX)||(this->size.nPixelsY != copyImage->getSize().nPixelsY)||(this->size.nPixelsZ != copyImage->getSize().nPixelsZ))
  {
	return false;
  }

  // Copio el contenido de la imagen.
  for(int i = 0; i < copyImage->nPixels; i++)
  {
	  this->pixels[i] = copyImage->pixels[i];
  }
  // Copio el resto de las propiedades:
  this->fovRadio_mm = copyImage->fovRadio_mm;
  this->fovHeight_mm = copyImage->fovHeight_mm;
  this->nPixels = copyImage->nPixels;
  
  return true;
}

bool Image::setSlice(int sliceIndex, Image* sliceToCopy)
{
    if((this->size.nPixelsZ <= sliceIndex)||(sliceToCopy->getSize().nPixelsZ != 1)||(this->size.nPixelsX != sliceToCopy->getSize().nPixelsX)
      ||(this->size.nPixelsY != sliceToCopy->getSize().nPixelsY))
    {
      //cout << "Error al copiar un slice por incompatibilidad de tamaños (Image::setSlice)" << endl;
      return false;
    }
    // Podría recorrer la imagen e ir copiando cada pixel. Pero directamente copio toda la emoria, es más fácil:
    int numPixelsSlice = this->size.nPixelsX * this->size.nPixelsY;
    int offsetSlice = sliceIndex*numPixelsSlice;
    
    memcpy(this->pixels+offsetSlice, sliceToCopy->getPixelsPtr(),numPixelsSlice*sizeof(float));
}

Image* Image::getSlice(int sliceIndex)
{
  SizeImage sizeSlice = this->getSize();
  sizeSlice.nPixelsZ = 1;
  Image* slice = new Image(sizeSlice);
  // Copio los píxeles ahora:
  for(int i = 0; i < sizeSlice.nPixelsX; i++)
  {
    for(int j = 0; j < sizeSlice.nPixelsY; j++)
    {
      slice->setPixelValue(i,j,0, this->getPixelValue(i,j,sliceIndex));
    }
  }
  return slice;
}

void Image::fillConstant(float value)
{
	for(unsigned int i = 0; i < this->nPixels; i++)
	{
		this->pixels[i] = value;
	}
}

// Method that writes the image data into a file.
bool Image::writeRawFile(char* filePath)
{
	FILE* fileImage = fopen(filePath,"wb");
	unsigned int CantBytes;
	if((CantBytes =  fwrite(this->pixels, sizeof(float), this->nPixels , fileImage)) !=  (this->nPixels))
		return false;
	fclose(fileImage);
	return true;
}

// Method that writes the image data into a file.
bool Image::writeInterfile(char* filePath)
{
  char *err, *tmpfname;
  FILEINFO* fi;	// Estructura que me describe un archivo
  fi = (FILEINFO*)malloc(sizeof(FILEINFO));
  // Inicialización del fi:
  MdcInitFI(fi, filePath);
  /*if(MdcOpenFile(fi, filePath)!=MDC_OK)
  {
    this->strError.assign("El archivo de salida ");
    this->strError += fi->ifname;
    this->strError += " no se pudo crear o abrir.";
    return false;
  }*/
  // Debo llenar el contenido de la estructura, para  generar el contenido interfile.
  sprintf(fi->manufacturer, "AR-PET") ;
  sprintf(fi->institution, "UTN-FRBA y CNEA");
  sprintf(fi->patient_name, "Imagen generada con APIRL");

  // En esta librería las imágenes 3D las consdiera como múltiples 2D, así que defino así
  // mi imagen.
  fi->number = this->size.nPixelsZ;
  fi->dim[0] = 3;	// 3 dimensiones
  fi->dim[1] = this->size.nPixelsX;
  fi->dim[2] = this->size.nPixelsY;
  fi->dim[3] = this->size.nPixelsZ;
  fi->pixdim[0] = 3;	// 3 dimnesiones
  fi->pixdim[1] = this->size.sizePixelX_mm;
  fi->pixdim[2] = this->size.sizePixelY_mm;
  fi->pixdim[3] = this->size.sizePixelZ_mm;
  // Tipo de adquisición, le ponemos tipo PET, que se lo agregué al código del medcon.
  // Revision, lo cambiamos a tipo TOMO, porque matlab no reconoce el tipo PET:
  fi->acquisition_type = MDC_ACQUISITION_PET;
  // Por ahora solo manejamos float:
  fi->type = FLT32;
  fi->endian = MDC_LITTLE_ENDIAN;
  // Pedimos memoria para las imagenes 2D:
  fi->image = (IMG_DATA*)malloc((sizeof(IMG_DATA)*fi->dim[3]));
  // Cargo la imagen en la estructura:
  int sizeSlice = this->size.nPixelsX * this->size.nPixelsY;
  for (int i=0; i<fi->number; i++)
  {
    fi->image[i].bits = MdcType2Bytes(fi->type);
    fi->image[i].height = fi->dim[2];
    fi->image[i].width = fi->dim[1];
    fi->image[i].type = fi->type;
    fi->image[i].pixel_xsize = fi->pixdim[1];
    fi->image[i].pixel_ysize = fi->pixdim[2];
    fi->image[i].buf = (Uint8*)(this->pixels + i*sizeSlice);

  }
  /// Pongo por default como imagen reconstruida. Se podr�a poner adquirida o darle la opci�n al m�todo, despu�s se puede modificar.
  fi->reconstructed == MDC_YES;

  // Debo
  // Escribo la imagen. Para eso primero creo el archivo, y lo abro antes de llamar a la función.
  // Para el archivo de datos le debo agregar la extensión i33, para el header h33:
  //tmpfname = (char*)malloc(strlen(fi->ifname));
  tmpfname = (char*)malloc(MDC_MAX_PATH);
  strcpy(fi->ofname,fi->ifname);
  MdcSetExt(fi->ofname,"i33");
  if((fi->ofp = fopen(fi->ofname,"wb")) == NULL)
  {
    strError.assign("Error al intentar crear el archivo de datos interfile.");
    return false;
  }
  // Ver estas funciones para ver como cargar bien la estructura, necesito definir algunas variables previamente:
  MDC_FILE_ENDIAN = MDC_LITTLE_ENDIAN;
  err = MdcWriteIntfImages(fi);
  if (err != NULL) {this->strError.assign(err); return false;}
  MdcCheckIntfDim(fi);
  MdcCloseFile(fi->ofp);

  // Ahora Escribo el header.
  strcpy(tmpfname,fi->ifname);
  // MdcNewExt me combina dos strings, pongo el de salida vacío porque sino me concatena el nombre que quedó del .i33
  // con el nombre base más el header.
  strcpy(fi->ofname, tmpfname);
  MdcSetExt(fi->ofname,"h33");
  if((fi->ofp = fopen(fi->ofname,"wb")) == NULL)
  {
    strError.assign("Error al intentar crear el archivo de encabezado interfile.");
    return false;
  }
  #define MDC_INTF_SUPPORT_DIALECT 1
  err = MdcWriteIntfHeader(fi);
  if (err != NULL) {this->strError.assign(err); return false;}
  MdcCloseFile(fi->ofp);
  free(fi->image);
  free(fi);
  free(err);
  return true;
}

// Method that reads the Image data from a file. 
bool Image::readFromFile(char* filePath)
{
  FILE* fileImage = fopen(filePath,"wb");
  unsigned int CantBytes;
  if((CantBytes =  fread(this->pixels, sizeof(float), this->nPixels, fileImage)) !=  this->nPixels)
    return false;
  fclose(fileImage);
  return true;
}


// Function that returns the raw data of the Michelogram
float* Image::getPixelsPtr()
{
	return pixels;
}

// Function that returns the raw data of the Michelogram
void Image::fromRawData(float* rawData)
{
  this->pixels = rawData;
}



// Method that reads the Image data from an Interfile image.
/** Ver de agregar códigos de errores.
	\todo Carga de imágenes interfile más complejas.
*/
bool Image::readFromInterfile(char* filePath)
{
  const char* msg = NULL;
  // Uso las funciones de la aplicaciónde libre distribución Medcon.
  // Ver después de embeber los métodos en una clase propia.
  FILEINFO* fi;	// Puntero a estructura del tipo FILEINFO definida en Medcon.
  int error;
  // Pido memoria para la estructura:
  fi = (FILEINFO*) malloc(sizeof(FILEINFO));
  // Abro el archivo en inicializo fi.
  if ((error = MdcOpenFile(fi, filePath)) != MDC_OK) return false;
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
  // Primero guardo los tamaños (solo acepto hasta 3 dimensiones):
  if(fi->dim[0] >= 2)
  {
    this->size.nPixelsX = fi->image[0].width;
    this->size.nPixelsY = fi->image[0].height;
    this->size.sizePixelX_mm = fi->image[0].pixel_xsize;
    /// También puedo tener la info en pixdim, especialmente para el caso de PET que agregué yo:
    if(fi->image[0].pixel_xsize == 0)
      this->size.sizePixelX_mm = fi->pixdim[1];
    this->size.sizePixelY_mm = fi->image[0].pixel_ysize;
    if(fi->image[0].pixel_ysize == 0)
      this->size.sizePixelY_mm = fi->pixdim[2];
    this->size.nPixelsZ = 1;
    
    if(fi->dim[0] == 3)
    {
	    // Agrego la tercera dimensión, si hay más dimensiones devuelvo un error.
	    this->size.nPixelsZ = fi->number;
	    if(this->size.nPixelsZ == 0)
		    this->size.nPixelsZ = fi->dim[3];
	    //this->size.sizePixelZ_mm = fi->image[0].slice_width;
	    //if(fi->image[0].slice_width == 0)
	      this->size.sizePixelZ_mm = fi->pixdim[3];
    }
    else if(fi->dim[0] > 3)
    {
	    this->strError = "No se soportan imágenes de más de 3 dimensiones";
	    return false;
    }

  }
  else
  {
    this->strError = "No existen imágenes de menos de 2 dimensiones";
    return false;
  }
  this->size.nDimensions = fi->dim[0];
  this->nPixels = this->size.nPixelsX * this->size.nPixelsY * this->size.nPixelsZ;


  // Ahora cargo los píxeles de la imagen. Esto depende del tipo de datos de la misma.
  // Por ahora solo soporto floats, en el futuro se convertirá a través de un template
  // en una clase que soporte todos los tiposde datos.

  // Pedir memoria. Lo hago con realloc porque en el constructor ya se pide memoria, pero
  // por otro lado debo pedirla ac� si o si porque en el archivo intefile me pueden
  // cambiar als dimensiones que se apsaron al constructor.
  this->pixels = (float*) realloc(this->pixels, sizeof(float)*this->nPixels);
  // Tipos de datos en medcon: BIT1, BIT8_U, BIT16_S, BIT32_S, FLT32, FLT64, COLRGB
  if(fi->type == FLT32)
  {
    int nPixelsSlice = this->size.nPixelsX * this->size.nPixelsY;
    IMG_DATA *id;
    for(int i = 0; i < fi->number; i++)
    {
      id = &fi->image[i];
      memcpy(this->pixels + i*nPixelsSlice, id->buf, sizeof(float)*nPixelsSlice);
    }
  }
  else
  {
    this->strError = "Al momento solo se soportan imágenes del tipo float.";
    return false;
  }
  
  
  // Cargo los valores de RFOV y ZFOV a partir de los valroes de p�xeles:
  this->fovRadio_mm = (this->size.sizePixelX_mm * this->size.nPixelsX)/2;
  this->fovHeight_mm = this->size.sizePixelZ_mm * this->size.nPixelsZ;
  return true;
}

int Image::forcePositive()
{
  int numNegatives = 0;
  for(int i = 0; i < this->nPixels; i++)
  {
    if(this->pixels[i] < 0)
    {
      numNegatives++;
      this->pixels[i] = 0;
    }
  }
  return numNegatives;
}

float Image::getMinValue()
{
  // Recorro todos los píxeles de la imagen. Busco el mínimo distinto de cero.
  float minValue = numeric_limits<float>::max();
  for(int i = 0; i < this->nPixels; i++)
  {
    if((this->pixels[i] != 0) && (minValue!=0))
    {
      if(this->pixels[i] < minValue)
      {
	minValue = this->pixels[i];
      }
    }
    else
    {
      if(this->pixels[i] != 0)
      {
	minValue = this->pixels[0];
      }
    }
  }
  return minValue;
}

float Image::getMaxValue()
{
  // Recorro todos los píxeles de la imagen.
  float maxValue = this->pixels[0];
  for(int i = 0; i < this->nPixels; i++)
  {
    if(this->pixels[i] > maxValue)
    {
      maxValue = this->pixels[i];
    }
  }
  return maxValue;
}

void Image::getPixelGeomCoord(int x, int y, float* x_mm, float* y_mm)
{
  // El extremo en x es -fovRadio_mm, le debo agregar la cantidad de píxeles más un medio para tener el centro del mismo.
  (*x_mm) = -this->fovRadio_mm + this->size.sizePixelX_mm *  ((float)x + 0.5); 
  // El extremo en y es +fovRadio_mm, le debo agregar la cantidad de píxeles más un medio para tener el centro del mismo.
  (*y_mm) = -this->fovRadio_mm + this->size.sizePixelY_mm *  ((float)y + 0.5); 
}

void Image::getPixelGeomCoord(int x, int y, int z, float* x_mm, float* y_mm, float* z_mm)
{
  // El extremo en x es -fovRadio_mm, le debo agregar la cantidad de píxeles más un medio para tener el centro del mismo.
  (*x_mm) = -this->fovRadio_mm + this->size.sizePixelX_mm *  ((float)x + 0.5); 
  // El extremo en y es +fovRadio_mm, le debo agregar la cantidad de píxeles más un medio para tener el centro del mismo.
  (*y_mm) = -this->fovRadio_mm + this->size.sizePixelY_mm *  ((float)y + 0.5); 
  // El extremo en z es fovHeight_mm, le debo agregar la cantidad de píxeles más un medio para tener el centro del mismo.
  //(*z_mm) = this->fovHeight_mm - this->size.sizePixelZ_mm *  ((float)z + 0.5);
  (*z_mm) = this->size.sizePixelZ_mm *  ((float)z + 0.5); 
}