/**
	\file Images.h
	\brief Archivo que contiene la definicion de la clase Image y la estructura SizeImage.

	Agregar mas detalles.
	\todo
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2010.08.30
	\version 1.0.0
*/

#ifndef _IMAGES_H
#define	_IMAGES_H
#include <string>
using namespace std;

// DLL export/import declaration: visibility of objects
#ifndef LINK_STATIC
	#ifdef WIN32               // Win32 build
		#pragma warning( disable: 4251 )
		#ifdef DLL_BUILD    // this applies to DLL building
			#define DLLEXPORT __declspec(dllexport)
		#else                   // this applies to DLL clients/users
			#define DLLEXPORT __declspec(dllimport)
		#endif
		#define DLLLOCAL        // not explicitly export-marked objects are local by default on Win32
	#else
		#ifdef HAVE_GCCVISIBILITYPATCH   // GCC 4.x and patched GCC 3.4 under Linux
			#define DLLEXPORT __attribute__ ((visibility("default")))
			#define DLLLOCAL __attribute__ ((visibility("hidden")))
		#else
			#define DLLEXPORT
			#define DLLLOCAL
		#endif
	#endif
#else                         // static linking
	#define DLLEXPORT
	#define DLLLOCAL
#endif



/**
	\brief Estructura que define el tamaño de una imagen.

	Esta estructura define el tamaño de una imagen, tanto en cantidad de dimensiones (2 o 3), tamaño
	en píxeles de la imagen para cada dimensión, y el tamaño en mm de cada píxel. De esta forma se
	tiene todo la información necesaria para definir el tamaño de una imagen. Esta estructura
	se utilizará en la clase Image y para pasaje de parámetros que requieran el tamaño de una imagen.
*/
struct DLLEXPORT SizeImage
{
	int nDimensions;
	int nPixelsX;	// Number of Pixels in X
	int nPixelsY;	// Number of Pixels in X
	int nPixelsZ;	// Number of Pixels in X
	float sizePixelX_mm;	// Number of Pixels in Y
	float sizePixelY_mm;	// Number of Pixels in Y
	float sizePixelZ_mm;	// Number of Pixels in Y
};

/**
	\brief La clase Image representa una imagen de 2 o 3 dimensiones.

	La clase Image representa una imagen que puede ser tanto de 2 o 3 dimensiones,
	según como sea definida. Al constructor se le pasa como parámetros la estructura
	SizeImage que define el tamaño de una imagen.

	\todo Una imagen genérica no debería tener fovRadio y fovLength, esto tiene más que ver con
		  imágenes médicas, se podría hacer una clase Image genérica y otra que sea PetImgae o
		  MedicalImage que derive de Image.
	\todo Hasta el momento la clase tiene la información de los píxeles en unn puntero público
		  del tipo float. Esto debería cambiarse para soporte distintos tipos de datos, por
		  ejemplo haciendo una clase píxeles que genérica y que tenga sus clases derviadas
		  FloatPixel, Uint8Pixel, etc. Otra forma sería con templates. Además para la obtención
		  de los píxeles se debería usar métodos getPixel y setPixel.
*/
class DLLEXPORT Image
{
public:
  	/// Constructor de la clase Image.
	/** Este constructor simplemente instancia el objeto, y no reaiza ninguna otra acción, no 
	    pide memoria, ni define ningún parámetro de la imagen. La existencia de este constructor
	    se justifica, cuando se quiere leer la imagen de un archivo. Primero se instancia la clase
	    y luego se llama a readFromFile o readFromInterfile.
	*/
	Image();
	
	/// Constructor de la clase Image.
	/** Este constructor simplemente instancia el objeto, y lo carga con una imagen interfile, cuyo
		nombre recibe como parámetro.
	*/
	Image(string imageFilename);
	
	/// Constructor de la clase Image.
	/** El constructor recibe como parámetro una estructura del tipo SizeImage donde se tienen todos los
		parámetros que hacen al tamaño de la imagen. En el constructor se asigna la memoria para los píxeles
		y se inicializa la iamgen en cero.
		\param sizeImage estructura del tipo SizeImage con el tamaño de imagen a instanciar.
	*/
	Image(SizeImage sizeImage);
	
	/// Constructor que genera una copia de una imagen representada en este objeto.
	/** Constructor que realiza la copia del objeto Image. Conveniente para cuando se
		desea tener una copia independiente de la iamgen.
		@param copyImage Puntero a Image donde guardará copia del objeto actual.
	*/
	Image(Image* copyImage);
	
	/// Destructor de la clase Image.
	/** El destructor libera la memoria utilizada por los píxeles.
	*/
	~Image();

	/// Método que guarda el valor de los píxeles en formato crudo.
	/** \param filePath path completo del archivo de datos crudos que se generará.
		\return false si hubo algún error, true si la operación fue exitosa.
	*/
	bool writeRawFile(char* filePath);
	
	/// Método que guarda el valor de los píxeles en formato interfile.
	/** \param filePath path completo del archivo del cual se generarán el archivo encabezado (.hv) y el imagen (.v).
		\return false si hubo algún error, true si la operación fue exitosa.
	*/
	bool writeInterfile(char* filePath);

	/// Método que llena todos los píxeles de la imagen con un valor constante.
	/** \param value valor que tomarán todos los píxeles de la imagen.
	*/
	void fillConstant(float value);

	/// Método que devuelve el puntero al array con los valores de los píxeles.
	/** 
		\return puntero del tipo float al array con los pixeles.
	*/
	float* getPixelsPtr();
	
	/// Método que lee los valores de los píxeles desde un archivo con datos crudos.
	/** \param filePath path completo del archivo de datos crudos que se leerá.
		\return false si hubo algún error, true si la operación fue exitosa.
	*/
	bool readFromFile(char* filePath);

	/// Método que lee una imagen del tipo intefile y la carga en este objeto..
	/** \param filePath path completo del archivo intefile que se leerá.
		\return false si hubo algún error, true si la operación fue exitosa.
	*/	
	bool readFromInterfile(char* filePath);

	/// Método que llena todos los píxeles de la imagen con un puntero a los píxeles de otra.
	/** \param rawData puntero a un float array con los valores de todos los píxeles de una imagen.
	*/
	void fromRawData(float* rawData);

	/// Método que obtiene la cantidad de píxeles totales que tiene la imagen.
	/** \return long int con la cantidad de píxeles totales de la imagen.
	*/
	long int getPixelCount() {return this->size.nPixelsX*this->size.nPixelsY*this->size.nPixelsZ;};
	
	/// Método que obtiene el valor de un píxel.
	/** Se debe pedir filla, columna, z.
		\return float valor del píxel pedido.
	*/
	float getPixelValue(int x, int y, int z) {return pixels[z * size.nPixelsY * size.nPixelsX +
	  y * size.nPixelsX + x];};
	  
	/// Método que asgina el valor de un píxel.
	/** Se debe indicar las coordenadas filla, columna y z del p.
		@param x coordenada x (columna) del píxel a actualizar.
		@param y coordenada y (fila) del píxel a actualizar.
		@param z coordenada z (altura) del píxel a actualizar.
		@param value valor que se le desea asginar al píxel.
		\return float valor del píxel pedido.
	*/
	float setPixelValue(int x, int y, int z, float value) { pixels[z * size.nPixelsY * size.nPixelsX +
	  y * size.nPixelsX + x] = value; return value;};
	  
	/// Método que asgina un slice de una imagen 3D con una imagen 2D recibida como parámetro.
	/** Se debe indicar que slice se desea asignar con una Imagen de dos dimensiones. Si
	 * no se cumple la condición que la imagen sea de 3 dimensiones y la imagen a asignar sea de dos
	 * devuelve false.
		@param sliceIndex coordenada x (columna) del píxel a actualizar.
		@param sliceToCopy imagen bidimensional que se aplicará al slice deseado.
		\return false si hubo algún error, true si la operación fue exitosa.
	*/
	bool setSlice(int sliceIndex, Image* sliceToCopy);
	
	/// Método que asgina un slice de una imagen 3D con una imagen 2D recibida como parámetro.
	/** Se debe indicar que slice se desea asignar con una Imagen de dos dimensiones. Si
	 * no se cumple la condición que la imagen sea de 3 dimensiones y la imagen a asignar sea de dos
	 * devuelve false.
		@param sliceIndex indice del slice (coordenada z) del slice a obtener.
		\return puntero a Image con la imagen de dos dimensiones con el slice correspondiente.
	*/
	Image* getSlice(int sliceIndex);
	
	/// Método que asgina el valor de un píxel.

	/** Se debe indicar las coordenadas filla, columna y z del p.

		@param x coordenada x (columna) del píxel a actualizar.

		@param y coordenada y (fila) del píxel a actualizar.

		@param z coordenada z (altura) del píxel a actualizar.

		@param value valor que se le desea asginar al píxel.

		\return float valor del píxel pedido.

	*/
	float incrementPixelValue(int x, int y, int z, float value) { pixels[z * size.nPixelsY * size.nPixelsX +
	  y * size.nPixelsX + x] += value; return pixels[z * size.nPixelsY * size.nPixelsX + y * size.nPixelsX + x];};
	  
	/// Método que obtiene el tamaño de una imagen.
	/** \return estructura del tipo SizeImage con todos los parámetros que hacen al tamaño de la imagen.
	*/
	SizeImage getSize() {return this->size;};
	
	/// Método que obtiene el radio del cilindro del FOV (Field of view).
	/** \return float con el radio del FOV.
	*/
	float getFovRadio() {return fovRadio_mm;};

	/// Método que obtiene la altura del cilindro del FOV (Field of view).
	/** \return float con la altura del FOV, correspondiente al tamaño en el eje z.
	*/
	float getFovHeight() {return fovHeight_mm;};
	
	/// Método que obtiene las coordenadas geométricas de un píxel en mm.
	/** Método que obtiene las coordenadas geométricas de un píxel en mm. El origen del sistema de coordenadas
		geométricos siempre se encuentra en el centro de la imagen, y en x crece en el mismo sentido que el índice
		de píxel, pero para y de forma inversa.
		@param x índice en x del píxel de la imagen.
		@param y índice en y del píxel de la imagen.
		@param z índice en z del píxel de la imagen.
		@param x_mm puntero a float donde se devolverá la coordenada geométrica x, en mm, del centro del píxel pedido.
		@param y_mm puntero a float donde se devolverá la coordenada geométrica y, en mm, del centro del píxel pedido.
		@param z_mm puntero a float donde se devolverá la coordenada geométrica <, en mm, del centro del píxel pedido.
	*/
	void getPixelGeomCoord(int x, int y, int z, float* x_mm, float* y_mm, float* z_mm);

	/// Método que obtiene las coordenadas geométricas de un píxel en mm para una iamgen 2d.
	/** Método que obtiene las coordenadas geométricas de un píxel en mm. El origen del sistema de coordenadas
		geométricos siempre se encuentra en el centro de la imagen, y en x crece en el mismo sentido que el índice
		de píxel, pero para y de forma inversa. Sobrecarga para imagen 2d.
		@param x índice en x del píxel de la imagen.
		@param y índice en y del píxel de la imagen.
		@param x_mm puntero a float donde se devolverá la coordenada geométrica x, en mm, del centro del píxel pedido.
		@param y_mm puntero a float donde se devolverá la coordenada geométrica y, en mm, del centro del píxel pedido.
	*/
	void getPixelGeomCoord(int x, int y, float* x_mm, float* y_mm);
	
	/// Método que obtiene un string con el error de algún mmétodo luego que devolvió false.
	/** \return false objeto del tipos tring con el mensaje de error del último error que apareció en el objeto.
	*/
	string getError() {return this->strError;};
	
	/// Método que obtiene el mínimo valor de píxel en la imagen.
	float getMinValue();
	
	/// Método que obtiene el máximo valor de píxel en la imagen.
	float getMaxValue();
	
	/// Copia los valores de una imagen, tiene que tener el mismo tamaño.
	bool CopyFromImage(Image* copyImage);
	
	/// Método que busca negativos y los fuerza a cero. Devuelve la cantidad de números negativos encontrados.
	int forcePositive();
	
protected:

private:
	/// Puntero a float con los valores de los píxeles.
	/** La información esta ordenada recorriendo la imagen en el siguiente orden: columna, fila, profundidad.
	    O sea, x, y, z. El primer píxel es el (0,0,0), (0,1,0)...,(0,N-1,0), (1,0,0).....(M-1, N-1, 0), (0,0,1),....
	*/
	float* pixels;

	/// Estructura del tipo SizeImage que define el tamaño de una iamgen.
	/**	Esta estructura me define todos los parámetros que hacen al tamaño 
		de la imagen, esto es el número de dimensiones, el tamaño de la imagen
		en píxeles, y el tamaño en mm de cada píxel.
	*/
	SizeImage size;

	/// Dimension del radio del FOV de la imagen.
	/** Este parámetro tiene sentido cuando es una imagen obtenida de un scanner
		con cierto FOV. El radio del mismo se considera como el ancho de la imagen
		que sería equivalente al tamaño en X, que debería ser igual al tamaño en Y
		ya que el FOV se considera cilíndrico, siendo la altura del cilindro el eje
		Z.
	*/
	float fovRadio_mm;		
	
	/// Dimension de la altura del FOV de la imagen.
	/** Este parámetro tiene sentido cuando es una imagen obtenida de un scanner
		con cierto FOV, y solo para imágenes de 3 dimensiones. Su valor es la altura
		del cilindro que es considerado como FOV, o sea es el tamaño en el eje Z de
		la imagen.
	*/
	float fovHeight_mm;
	
	/// Cantidad de píxeles que contiene la imagen.
	long int nPixels;

	/// String de error.
	/** Objeto del tipo string donde se guarda un mensaje de error cuando esto ocurre en 
		algún método de esta clase.
	*/
	string strError;
};


#endif
