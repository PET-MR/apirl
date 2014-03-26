/**
	\file generateImage.cpp
	\brief Archivo de código para el comando generateImage

	Este archivo define el comando generateImage. Este comando recibe como argumento un archivo *.par
	que define los parámetros de la imagen a generar. El archivo de parámetros es el propuesto por STIR
	pero con algunas funcionalidades menos. Al momento existen dos tipos de imágenes que se pueden generar:
	cuadrados uniforme o cilindros uniformes. El tamaño de la imagen y de los píxeles se pasa a través del 
	archivo de parámetros.
	
	\par Ejemplo de Parametros.par
	\code
		generateImage Parameters :=
		output filename := image
		; optional keyword to specify the output file format
		; example below uses Interfile with 16-bit unsigned integers
		output file format type := Interfile
		  interfile Output File Format Parameters:=
			number format := unsigned integer
			number_of_bytes_per_pixel:= 2
			; fix the scale factor to 1
			; comment out next line to let STIR use the full dynamic 
			; range of the output type
			scale_to_write_data := 1
		  End Interfile Output File Format Parameters :=

		X output image size (in pixels) := 128
		Y output image size (in pixels) := 128
		Z output image size (in pixels) := 95
		X voxel size (in mm) := 2.05941
		Y voxel size (in mm) := 2.05941
		Z voxel size (in mm) := 2.425

		; parameters that determine subsampling of border voxels
		; to obtain smooth edges
		; setting these to 1 will just check if the centre of the voxel is in or out
		; default to 5
		; Z number of samples to take per voxel := 5
		; Y number of samples to take per voxel := 5
		; X number of samples to take per voxel := 5
			    
		shape type:= cylinder
		Cylinder Parameters :=
		   radius (in mm) := 100
		   length-z (in mm) := 400
		   ; next keyword can be used for non-default axes
		   ; values below are give a rotation around y for 90 degrees (swapping x and z)
		   ; Warning: this uses the STIR convention {z,y,x}
		   ; direction vectors (in mm):= { {0,0,1}, {0,1,0}, {-1,0,0}}
		   ; origin w.r.t. to standard STIR coordinate system (middle of first plane)
		   origin (in mm):={230.375, 3.0192, -0.590588}
		   END :=
		value := 1

		; next shape :=
		; see Shape3D hierarhcy for possibly shapes

		END :=
	\endcode
	\bug  El y positivo geométrico debe ir para arriba. Pendiente apra corregir. Sacar esto cuando se haga.

	\todo Se podría agregar más formas.
	\todo Hacer todo más modular orientado a objetos, esto es utilizar una clase o estructura GenImageParameters, donde se tienen
		  todos los datos de configuración. Además se crearía una clase abstracta shape, con todas las operaciones que debería
		  tener cada shape, y luego clases derivadas para cada shape aceptada. Por ahora no tengo tiempo para esto, y es algo accesorio.
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2010.09.09
	\version 1.0.0
*/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <string>
#include <iostream>
#include <ParametersFile.h>
#include <Images.h>
#include <Geometry.h>
using	namespace std;
using	std::string;

//#include <generateImage.h>

/**
	\fn void main (int argc, char *argv[])
	\brief Ejecutable que genera una imagen. Para eso recibe como parámetros el nombre del archivo .par de configuración de parámetros de la imagen.
	
	Este comando genera una imagen artificial a partir de los parámetros recibidos en el archivo .par.
	Cada entrada debe estar formado por la sentencia "Campo := Valor", respetando un único espacio en blanco
	a cada lado del :=.
	Los tipos de imágenes disponibles al momento son:
		- Imagen Uniforme: Imagen con un valor de píxel constante.
		- Cilindro: Cilindro (XY circular, y en Z el alrgo) cuyo interior los píxeles tienen un valor uniforme y el resto en cero.
	Parámetro del archivo *.par:
		- "output filename" : Nombre del archivo de salida (Sin extensión que son .hv para el header y .v para los datos).
		- "output file format type" : Formato del archivo de salida, por ahora solo disponible interfile y raw. Cada formato tiene
									  sus respectivos subparametros. Para interfile:
										"number format" : formato del píxel (unsigned integer, integer, double, float, etc)
										"number_of_bytes_per_pixel" : bytes por píxel
										"scale_to_write_data" : parámetros opcional para escalar el valor de píxeles entre 0 y 1
		- "X output image size (in pixels)" : tamaño de la imagen en píxeles en X.
		- "Y output image size (in pixels)" : tamaño de la imagen en píxeles en Y.
		- "Z output image size (in pixels)" : tamaño de la imagen en píxeles en Z.
		- "X voxel size (in mm)" : tamaño del píxel en X. Estos parámetros me determinan el tamaño del FOV de esta imagen.
		- "Y voxel size (in mm)" : tamaño del píxel en Y.
		- "Z voxel size (in mm)" : tamaño del píxel en Z.
		- "shape type" : Tipo de geometría a generar con la imagen. Por ahora, solo cylinder y box.
						 Subparámetros para cylinder:
							"radius (in mm)" : radio en mm.
							"length-z (in mm)" : largo del cilindro (en eje z).
							"origin (in mm)" : centro de la figura como "{X,Y,Z}" respecto del sistema de coordenadas geométrico. Cuyo origen está en 
											   el medio de la imagen, o sea X0 = SizeX*SizPixelX/2.
					     Subparámetros para box:
							"length-x (in mm)" : lado en mm en X.
							"length-y (in mm)" : lado en mm en Y.
							"length-z (in mm)" : lado en mm en Z.
							"origin (in mm)" : centro de la figura como "{X,Y,Z}" respecto del sistema de coordenadas geométrico. Cuyo origen está en 
											   el medio de la imagen, o sea X0 = SizeX*SizPixelX/2.
			    
 * @param argc Cantidad de argumentos de entrada
 * @param argv Puntero a vector con los argumentos de entrada. El comando debe ejecutarse con el nombre del archivo de parámetros como argumento.
 * @return 0 si no uhbo errores, 1  si falló la operación.
 */

int main (int argc, char *argv[]) 
{
	// Comando generateImage
	short int  i, pos;
	short int  count=0;    /* counter: How often appears keyword in the header? */
	int        n;
	char       *c[1];
	char errorMessage[300];	// string de error para la función de lectura de archivo de parámetros.
	char returnValue[256];	// string en el que se recibe el valor de un keyword en la lectura del archivo de parámetros.
	///char       line[512];  /* max length of a line accepted in interfile header */
	char	   firstLine[512]; /* línea inicial del archivo */
	string parameterFileName;	// string para el Nombre de Archivo de parámetros.
	int	errorCode;
	char* line = (char*)malloc(512);
	/// Inicializo los char[] porque me traen algún error en VS modo Debug.
	strcpy(line, "");
	strcpy(firstLine, "");
	memset(line, '\0', 512);
	memset(firstLine, '\0', 512);
	memset(returnValue, '\0', 256);
	memset(errorMessage, '\0', 300);


	string interfileHeader;	// string para el Nombre del archivo de header de la imagen.
	string outputFileName;	// Nombre del archivo de salida.

	// Objeto Imagen:
	Image* image;
	SizeImage sizeImage;
	float* ptrPixels;	// Puntero a los píxeles.

	//FILE       *;
	
	// Verificación de que se llamo al comando con el nombre de archivo de parámetros como argumento.
	if(argc != 2)
	{
		cout << "El comando generateImage debe llamarse indicando el archivo de Parámetros: generateImage Param.par." << endl;
		return -1;
	}

	// Los parámetros de reconstrucción son los correctos.
	// Se verifica que el archivo tenga la extensión .par.
	parameterFileName.assign(argv[1]);
	//strcpy(parameterFileName, argv[1]);
	if(parameterFileName.compare(parameterFileName.length()-4, 4, ".par"))
	{
		// El archivo de parámetros no tiene la extensión .par.
		cout<<"El archivo de parámetros no tiene la extensión .par."<<endl;
		return -1;
	}
	// Leo cada uno de los campos del archivo de parámetros.
	if((errorCode=parametersFile_read((char*)parameterFileName.c_str(), "generateImage", "output filename", (char*)returnValue, (char*)errorMessage)) != 0)
	{
		// Hubo un error. Salgo del comando.
		cout<<"Error "<<errorCode<<" en el archivo de parámetros. Mirar la documentación de los códigos de errores."<<endl;
		return -1;
	}
	outputFileName.assign(returnValue);
    // Cargo los parámetros de la imagen en la estructura del tipo SizeImage.
	// Píxeles en X.
	if((errorCode=parametersFile_read((char*)parameterFileName.c_str(), "generateImage", "X output image size (in pixels)", returnValue, errorMessage)) != 0)
	{
		// Hubo un error. Salgo del comando.
		cout<<"Error "<<errorCode<<" en el archivo de parámetros. Mirar la documentación de los códigos de errores."<<endl;
		return -1;
	}
	sizeImage.nPixelsX = atoi(returnValue);
	// Píxeles en Y.
	if((errorCode=parametersFile_read((char*)parameterFileName.c_str(), "generateImage", "Y output image size (in pixels)", returnValue, errorMessage)) != 0)
	{
		// Hubo un error. Salgo del comando.
		cout<<"Error "<<errorCode<<" en el archivo de parámetros. Mirar la documentación de los códigos de errores."<<endl;
		return -1;
	}
	sizeImage.nPixelsY = atoi(returnValue);
	// Píxeles en Z.
	if((errorCode=parametersFile_read((char*)parameterFileName.c_str(), "generateImage", "Z output image size (in pixels)", returnValue, errorMessage)) != 0)
	{
		// Hubo un error. Salgo del comando.
		cout<<"Error "<<errorCode<<" en el archivo de parámetros. Mirar la documentación de los códigos de errores."<<endl;
		return -1;
	}
	sizeImage.nPixelsZ = atoi(returnValue);
	// Tamaño de Píxeles en X.
	if((errorCode=parametersFile_read((char*)parameterFileName.c_str(), "generateImage", "X voxel size (in mm)", returnValue, errorMessage)) != 0)
	{
		// Hubo un error. Salgo del comando.
		cout<<"Error "<<errorCode<<" en el archivo de parámetros. Mirar la documentación de los códigos de errores."<<endl;
		return -1;
	}
	sizeImage.sizePixelX_mm = atof(returnValue);
	// Tamaño de Píxeles en Y.
	if((errorCode=parametersFile_read((char*)parameterFileName.c_str(), "generateImage", "Y voxel size (in mm)", returnValue, errorMessage)) != 0)
	{
		// Hubo un error. Salgo del comando.
		cout<<"Error "<<errorCode<<" en el archivo de parámetros. Mirar la documentación de los códigos de errores."<<endl;
		return -1;
	}
	sizeImage.sizePixelY_mm = atof(returnValue);
	// Tamaño de Píxeles en Z.
	if((errorCode=parametersFile_read((char*)parameterFileName.c_str(), "generateImage", "Z voxel size (in mm)", returnValue, errorMessage)) != 0)
	{
		// Hubo un error. Salgo del comando.
		cout<<"Error "<<errorCode<<" en el archivo de parámetros. Mirar la documentación de los códigos de errores."<<endl;
		return -1;
	}
	sizeImage.sizePixelZ_mm = atof(returnValue);

	// Ahora genero la imagen incial.
	image = new Image(sizeImage);
	ptrPixels = image->getPixelsPtr();

	// Leo el tipo de forma de la imagen.
	if((errorCode=parametersFile_read((char*)parameterFileName.c_str(), "generateImage", "shape type", returnValue, errorMessage)) != 0)
	{
		// Hubo un error. Salgo del comando.
		cout<<"Error "<<errorCode<<" en el archivo de parámetros. Mirar la documentación de los códigos de errores."<<endl;
		return -1;
	}
	if(!strcmp(returnValue, "cylinder"))
	{
		// Es un cilindro
		float radius_mm, lengthZ_mm, value;
		Point3D origin_mm;
		if((errorCode=parametersFile_read((char*)parameterFileName.c_str(), "generateImage", "radius (in mm)", returnValue, errorMessage)) != 0)
		{
			// Hubo un error. Salgo del comando.
			cout<<"Error "<<errorCode<<" en el archivo de parámetros. Mirar la documentación de los códigos de errores."<<endl;
			return -1;
		}
		radius_mm = atof(returnValue);
		if((errorCode=parametersFile_read((char*)parameterFileName.c_str(), "generateImage", "length-z (in mm)", returnValue, errorMessage)) != 0)
		{
			// Hubo un error. Salgo del comando.
			cout<<"Error "<<errorCode<<" en el archivo de parámetros. Mirar la documentación de los códigos de errores."<<endl;
			return -1;
		}
		lengthZ_mm = atof(returnValue);
		if((errorCode=parametersFile_read((char*)parameterFileName.c_str(), "generateImage", "value", returnValue, errorMessage)) != 0)
		{
			// Hubo un error. Salgo del comando.
			cout<<"Error "<<errorCode<<" en el archivo de parámetros. Mirar la documentación de los códigos de errores."<<endl;
			return -1;
		}
		value = atof(returnValue);
		if((errorCode=parametersFile_read((char*)parameterFileName.c_str(), "generateImage", "origin (in mm)", returnValue, errorMessage)) != 0)
		{
			// Hubo un error. Salgo del comando.
			cout<<"Error "<<errorCode<<" en el archivo de parámetros. Mirar la documentación de los códigos de errores."<<endl;
			return -1;
		}
		// Tengo {x,y,z}
		// Primero elimino las dos llaves:
		string aux = returnValue;
		aux = aux.substr(1, aux.length()-2);
		// Ahora me voy quedando con los números
		int pos = aux.find(",",0);
		string num = aux.substr(0,pos-1);
		// Convierto a numero.
		origin_mm.X = atof(num.c_str());
		// Segunda coordenada.
		int pos2 = aux.find(",", pos+1);
		num = aux.substr(pos+1, pos2-1);
		origin_mm.Y = atof(num.c_str());
		// Tercera coordenada
		pos = aux.find(",", pos2+1);
		num = aux.substr(pos2+1, pos-1);
		origin_mm.Z = atof(num.c_str());

		// Ahora genero la imagen
		ptrPixels = image->getPixelsPtr();
		long int pixelsSlice = sizeImage.nPixelsX * sizeImage.nPixelsY;
		float x_mm = 0, y_mm = 0, z_mm = 0;
		for(int k = 0; k < sizeImage.nPixelsZ; k++)
		{
			for(int j = 0; j < sizeImage.nPixelsY; j++)
			{
				for(int i = 0; i < sizeImage.nPixelsX; i++)
				{
					// Fuerzo a cero al pixel.
					ptrPixels[k*pixelsSlice + j * sizeImage.nPixelsX + i] = 0;
					// Si el pixel queda dentro del cilindro le asigno el valor de value:
					// Primero verifico el valor de z:
					z_mm = k * sizeImage.sizePixelZ_mm + sizeImage.sizePixelZ_mm/2;
					if((z_mm>(origin_mm.Z - lengthZ_mm/2))&&(z_mm<(origin_mm.Z + lengthZ_mm/2)))
					{
						// La coordenada Z está dentro, ahora verifico si la XY da dentro del circulo.
						// Ahora el centro está en el centro del plano XY. Cuando la imagen tiene cantidad
						// de pixeles pares en un sentido, significa que ningún píxel va a estar en la 
						// coordenada (0,0) sino que van a quedar -sizePixel/2 y + sizePixel/2.
						// x_mm = (i-sizeImage.nPixelsX/2) * sizeImage.sizePixelX_mm + (sizeImage.sizePixelX_mm\2)
						// Si la cantidad de píxeles es impar, si voy a tener un pixel centrado en (0,0)
						// x_mm = (i-sizeImage.nPixelsX/2) * sizeImage.sizePixelX_mm .
						// Puedo meter las dos condiciones, de la siguiente manera:
						// x_mm = (i-(int)(sizeImage.nPixelsX/2)) * sizeImage.sizePixelX_mm + (sizeImage.nPixelsX%2);
						x_mm = (i-(int)(sizeImage.nPixelsX/2)) * sizeImage.sizePixelX_mm + (1-sizeImage.nPixelsX%2) * sizeImage.sizePixelX_mm/2;
						y_mm = ((int)(sizeImage.nPixelsY/2)-j) * sizeImage.sizePixelY_mm + (1-sizeImage.nPixelsY%2) * sizeImage.sizePixelY_mm/2;
						if(((x_mm-origin_mm.X)*(x_mm-origin_mm.X) + (y_mm-origin_mm.Y)*(y_mm-origin_mm.Y)) < (radius_mm*radius_mm))
						{
							// Estoy dentro del cilindro!
							ptrPixels[k*pixelsSlice + j * sizeImage.nPixelsX + i] = value;
						}
					}
				}
			}
		}
	}
	else if(!strcmp(returnValue, "box"))
	{
		// Es un cilindro
		float lengthX_mm, lengthY_mm, lengthZ_mm, value;
		Point3D origin_mm;
		if((errorCode=parametersFile_read((char*)parameterFileName.c_str(), "generateImage", "length-x (in mm)", returnValue, errorMessage)) != 0)
		{
			// Hubo un error. Salgo del comando.
			cout<<"Error "<<errorCode<<" en el archivo de parámetros. Mirar la documentación de los códigos de errores."<<endl;
			return -1;
		}
		lengthX_mm = atof(returnValue);
		if((errorCode=parametersFile_read((char*)parameterFileName.c_str(), "generateImage", "length-y (in mm)", returnValue, errorMessage)) != 0)
		{
			// Hubo un error. Salgo del comando.
			cout<<"Error "<<errorCode<<" en el archivo de parámetros. Mirar la documentación de los códigos de errores."<<endl;
			return -1;
		}
		lengthY_mm = atof(returnValue);
		if((errorCode=parametersFile_read((char*)parameterFileName.c_str(), "generateImage", "length-z (in mm)", returnValue, errorMessage)) != 0)
		{
			// Hubo un error. Salgo del comando.
			cout<<"Error "<<errorCode<<" en el archivo de parámetros. Mirar la documentación de los códigos de errores."<<endl;
			return -1;
		}
		lengthZ_mm = atof(returnValue);
		if((errorCode=parametersFile_read((char*)parameterFileName.c_str(), "generateImage", "value", returnValue, errorMessage)) != 0)
		{
			// Hubo un error. Salgo del comando.
			cout<<"Error "<<errorCode<<" en el archivo de parámetros. Mirar la documentación de los códigos de errores."<<endl;
			return -1;
		}
		value = atof(returnValue);
		if((errorCode=parametersFile_read((char*)parameterFileName.c_str(), "generateImage", "origin (in mm)", returnValue, errorMessage)) != 0)
		{
			// Hubo un error. Salgo del comando.
			cout<<"Error "<<errorCode<<" en el archivo de parámetros. Mirar la documentación de los códigos de errores."<<endl;
			return -1;
		}
		// Tengo {x,y,z}
		// Primero elimino las dos llaves:
		string aux = returnValue;
		aux = aux.substr(1, aux.length()-2);
		// Ahora me voy quedando con los números
		int pos = aux.find(",",0);
		string num = aux.substr(0,pos-1);
		// Convierto a numero.
		origin_mm.X = atof(num.c_str());
		// Segunda coordenada.
		int pos2 = aux.find(",", pos+1);
		num = aux.substr(pos+1, pos2-1);
		origin_mm.Y = atof(num.c_str());
		// Tercera coordenada
		pos = aux.find(",", pos2+1);
		num = aux.substr(pos2+1, pos-1);
		origin_mm.Z = atof(num.c_str());

		// Ahora genero la imagen
		ptrPixels = image->getPixelsPtr();
		long int pixelsSlice = sizeImage.nPixelsX * sizeImage.nPixelsY;
		float x_mm = 0, y_mm = 0, z_mm = 0;
		for(int k = 0; k < sizeImage.nPixelsZ; k++)
		{
			for(int j = 0; j < sizeImage.nPixelsY; j++)
			{
				for(int i = 0; i < sizeImage.nPixelsX; i++)
				{
					// Fuerzo a cero al pixel.
					ptrPixels[k*pixelsSlice + j * sizeImage.nPixelsX + i] = 0;
					// Si el pixel queda dentro del cilindro le asigno el valor de value:
					// Primero verifico el valor de z:
					z_mm = k * sizeImage.sizePixelZ_mm + sizeImage.sizePixelZ_mm/2;
					if((z_mm>(origin_mm.Z - lengthZ_mm/2))&&(z_mm<(origin_mm.Z + lengthZ_mm/2)))
					{
						// La coordenada Z está dentro, ahora verifico si la XY da dentro del ccuadrado.
						// Ahora el centro está en el centro del plano XY. Cuando la imagen tiene cantidad
						// de pixeles pares en un sentido, significa que ningún píxel va a estar en la 
						// coordenada (0,0) sino que van a quedar -sizePixel/2 y + sizePixel/2.
						// x_mm = (i-sizeImage.nPixelsX/2) * sizeImage.sizePixelX_mm + (sizeImage.sizePixelX_mm\2)
						// Si la cantidad de píxeles es impar, si voy a tener un pixel centrado en (0,0)
						// x_mm = (i-sizeImage.nPixelsX/2) * sizeImage.sizePixelX_mm .
						// Puedo meter las dos condiciones, de la siguiente manera:
						// x_mm = (i-(int)(sizeImage.nPixelsX/2)) * sizeImage.sizePixelX_mm + (sizeImage.nPixelsX%2);
						x_mm = (i-(int)(sizeImage.nPixelsX/2)) * sizeImage.sizePixelX_mm + (1-sizeImage.nPixelsX%2) * sizeImage.sizePixelX_mm/2;
						y_mm = ((int)(sizeImage.nPixelsY/2)-j) * sizeImage.sizePixelY_mm + (1-sizeImage.nPixelsY%2) * sizeImage.sizePixelY_mm/2;
						if((abs(x_mm-origin_mm.X)<(lengthX_mm/2)) && (abs(y_mm-origin_mm.Y)<(lengthY_mm/2)))
						{
							// Estoy dentro del cilindro!
							ptrPixels[k*pixelsSlice + j * sizeImage.nPixelsX + i] = value;
						}
					}
				}
			}
		}
	}
	else
	{
		// Hubo un error. Salgo del comando.
		cout<<"Error: al momento solo se soporta el shape del tipo cylinder y box."<<endl;
		return -1;
	}
	// Si llegué acá es porque generé la iamgen.
	// La guardo en el formato pedido, por ahora solo raw e interfile:
	if((errorCode=parametersFile_read((char*)parameterFileName.c_str(), "generateImage", "output file format type", returnValue, errorMessage)) != 0)
	{
		// Hubo un error. Salgo del comando.
		cout<<"Error "<<errorCode<<"en el archivo de parámetros. Mirar la documentación de los códigos de errores."<<endl;
	}
	if(!strcmp(returnValue, "Interfile"))
	{
		// Formato interfile
		/*
		"number format" : formato del píxel (unsigned integer, integer, double, float, etc)
		"number_of_bytes_per_pixel" : bytes por píxel
		"scale_to_write_data" : parámetros opcional para escalar el valor de píxeles entre 0 y 1
		*/
		if(image->writeInterfile((char*)outputFileName.c_str()))
		  cout<<"Imagen creada en formato Interfile."<<endl;
		else
		  cout<<"No se pudo generar la imagen: "<<image->getError()<<"."<<endl;
	}
	else if(!strcmp(returnValue, "raw"))
	{
		// Datos crudos:
		image->writeRawFile((char*)outputFileName.c_str());
		cout<<"Imagen creada en formato Raw.";
	}


	
}
