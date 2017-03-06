/**
	\file ParametersFile.cpp
	\brief Archivo para la lectura de archivos de parmetros que forma parte de la shared library "data"

	Este archivo define la funcion parametersFile_read que permite leer los campos del archivo de configuración
	de parámetros necesario para los distintos comandos de la biblioteca. El archivo de parámetros tiene un estilo similar
	a los archivos interfiles, diferenciándose en el cabezado. En el cual en vez de la palabra clave !INTERFILE,
	se encuentra la palabra clave del comando correspondiente. Al momento estas son:
		MLEM Parameters
		generateImage Parameters
	
	\todo Se podría generar una clase más genérica.
	      Solo hay implementada una función que lee un campo y devuelve el valor. Están definidas
		  dos funciones para leer un set de keys o para leer tun subgrupo de keys, las 
		  mismas falta ser codificadas.
	      Reemplazar los códigos de error en número por los define definidos en ParametersFile.h.
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2010.09.03
	\version 1.1.0
*/

#include <stdio.h>
#include <string.h>
#include <ParametersFile.h>

/*!
 * La función busca las palabras claves de un campo del archivo de parámetros y 
 * devuelve el valor correspondiente a ese campo.
 * Para esto es necesario pasarle como parámetros el nombre del archivo de parámetros,
 * el tipo de parámetros del archivo (depende del comando, por lo gral se lo identifica directamente con el nombre del comando)
 * y el nombre campo que se desea buscar. El valor del campo es devuelto en un string
 * por lo que la interpretación del campo deberá hacerse desde el programa que se llama
 * a esta función.
 *
 * El archivo de parámetros tiene que cumplir con las siguientes reglas:
 * - la primera línea es 'COMANDO Parameters :=', dependiendo del comando a parametrizar. Al momento: 'MLEM Parameter', 'GenerateImage Parameter'
 * - el máximo largo por línea es de 512 caracteres.
 * - una línea tiene dos strins separados por ':=' (keyword := valor) y con un espacio a cada lado del ':='.
 * - el máximo largo de cada keyword y valor es de 256 caracteres.
 * - se da por finalizado el archivo con la línea 'END:'
 * - una línea iniciando con punto y coma ';' es un comentario
 * - Existen subgrupos de parámetros que por ahora no son tratados como subgrupos, por lo que los espacios
 *   o '\t' al principio de un campo de un subgrupo son eliminados y se los considera como una entrada común. 
 *
 * @param fileName Nombre del archivo de parámetros de reconstrucción
 * @param parameterType Nombre del tipo de parámetros a definir (Nombre del comando, por lo general).
 * @param searchWord keyword a buscar
 * @param returnValue valor para esa keyword 
 * @param errorMessage mensaje de error o warning. 
 * @return 0 si no uhbo errores, 1  si la keyword aparece más de una vez en el archivo,
 * 2  si la keyword no fue encontrada (returned value is empty (i.e. contains '/0's only)),
 * 3  el archivo no pude abrirse para lectura (returned value is empty (i.e. contains '/0's only)),
 * 4  Posiblemente el archivo tiene otro formato (No se encontró el METODOParameters al inicio, o ese método no esta disponible) (returned value is empty (i.e. contains '/0's only)),
 * 5  el método buscado no es válido.
 */
int parametersFile_read(char* fileName, char* parameterType, char* searchWord, char* returnValue, char* errorMessage) {
  short int  i, pos, pos0;
  short int  count=0;    /* counter: How often appears keyword in the header? */
  int        n;
  char       *c[1];
  char       keyword[256], value[256];
  char       line[520];  /* max length of a line accepted in interfile header */
  char		 firstLine[512]; /* línea inicial del archivo */
  FILE       *interfileHeader;

  /// Inicialización sino puede generar algún error.
  /* initialise strings */
  for (i=0;i<256;i++) returnValue[i] = '\0';
  for (i=0;i<300;i++) errorMessage[i] = '\0';
  for (i=0;i<256;i++) keyword[i] = '\0';
  for (i=0;i<256;i++) value[i] = '\0';
  for (i=0;i<512;i++) firstLine[i] = '\0';
  for (i=0;i<520;i++) line[i] = '\0';

                                         /* open interfile header for reading */
  if ((interfileHeader = fopen(fileName,"r"))==NULL) {
    strcpy(errorMessage,fileName);
    strcat(errorMessage," could not be opened for reading");
    return 3;
  }

                  /* check from first line if file is really interfile header */
  n=(int)fread(&c,1,1,interfileHeader); if(n<1) {
    strcpy(errorMessage,"wrong file header format?! No Header at start of ");
    strcat(errorMessage,fileName);
    fclose(interfileHeader);
    return 4;
  }
  i=0;
  memcpy(&line[i],c,1);
  while (memcmp(c,"\n",1) && memcmp(c,"\r",1)) {
    i++;
    n=(int)fread(&c,1,1,interfileHeader); if(n<1) {
		strcpy(errorMessage,"wrong file header format?! No 'Method Parameters :=' at start of ");
      strcat(errorMessage,fileName);
      fclose(interfileHeader);
      return 4;
    }
    memcpy(&line[i],c,1);
  }
  // Concateno método de reconstrucción con "Parameter :=" para formar la primera línea que tiene que tener el archivo.
  strcpy(firstLine, parameterType);
  strcat(firstLine, " Parameters :=");
  //if (memcmp(line,"MLEMParameter :=",16)&&memcmp(line,"OSEMParameter :=",16)) {
  if (memcmp(line,firstLine,strlen(firstLine))) {
    strcpy(errorMessage,"wrong file header format?! No 'Method Parameters :=' or the Method is not valid at start of ");
    strcat(errorMessage,fileName);
    fclose(interfileHeader);
    return 5;
  }

                                                    /* read file line by line */
 while (fread(&c,1,1,interfileHeader) == 1) {
    for (i=0;i<516;i++) line[i] = '\0';                    /* initialise line */
    for (i=0;i<256;i++) keyword[i] = '\0';              /* initialise keyword */
    for (i=0;i<256;i++) value[i] = '\0';                  /* initialise value */
    i=0;
             /* \n = end of line, \r = carriage return. Lines in  ASCII files */
             /* on Sun-Solaris end with \n, on Intel-Windows with \r\n        */
    while (memcmp(c,"\r",1) && memcmp(c,"\n",1) && i<516) {
      memcpy(&line[i],c,1);
      n=(int)fread(&c,1,1,interfileHeader); if(n<1) {
        strcpy(errorMessage,"wrong file header format: ");
        strcat(errorMessage,fileName);
        fclose(interfileHeader);
        return 4;
      }
      i++;
    }
                                                /* comments are not processed */
    if (strncmp(&line[0],";",1)) {
                                           /* get keyword and value from line */
                                 /* find position of the field seperator ':=' */
      for (pos=1; pos<516; pos++)
        if (line[pos] == '=' && line[pos-1] == ':') break; 
                                    /* now get the first and the second field */
      /* Agregado por Martín B. para no considerar los espacios en blanco
      al inicio de una línea como caracteres válidos: */
      for (i=0;i<pos-2 && i<256;i++)
      {
	// Caundo llego a un caracter distinto del espacio en blanco ' ' o el '\t' 
	// considero que incia la línea ahí
	if(memcmp(&line[i],"\t",1) && memcmp(&line[i]," ",1))
	{
	  pos0 = i;
	  break;
	}
      }
      /* Fin Agregado*/
      /* Modifico la siguiente linea consistentemente con lo agregado atrás:
      for (i=0;i<pos-2 && i<256;i++) keyword[i] = line[i];
      */
      for (i=pos0;i<pos-2 && i<256;i++) keyword[i-pos0] = line[i];
      for (i=pos+2;i<256+pos+2 && i<512;i++) {
        if (!memcmp(&line[i],"\0",1) || !memcmp(&line[i],"\r",1) || !memcmp(&line[i],"\n",1)) 
          break;                                 /* stop at the end of "line" */
        value[i-pos-2] = line[i];
      }
	  if (!memcmp(keyword,"END :=",6)) break;     /* are we done? */
                                             /* check if we found the keyword */
       else if (!strcmp(keyword,searchWord)) {
              strcpy(returnValue,value);
              count++;
            }
    }
  }
  fclose(interfileHeader);                               /* done with reading */
  if (count == 0) {
    strcpy(errorMessage,"keyword '");
    strcat(errorMessage,searchWord);
    strcat(errorMessage,"' not found in header");
    return 2;
  }
  if (count > 1) {
    strcpy(errorMessage,"keyword '");
    strcat(errorMessage,searchWord);
    strcat(errorMessage,"' appears more than once in header");
    return 1;
  }
  return 0;
}


/*!
 * La función busca múltiples palabras claves de campos del archivo de parámetros y 
 * devuelve los valores correspondiente a cada campo.
 * Para esto es necesario pasarle como parámetros el nombre del archivo de parámetros,
 * el tipo de parámetros del archivo (depende del comando, por lo gral se lo identifica directamente con el nombre del comando)
 * y los nombres de los campos que se desea buscar. Los valores de los campos
 * son devueltos en un array de strings, o sea un doble puntero a char*.
 * La interpretación del campo deberá hacerse desde el programa que se llama
 * a esta función.
 *
 * El archivo de parámetros de reconstrucción tiene que cumplir con las siguientes reglas:
 * - la primera línea es 'COMANDO Parameters :=', dependiendo del comando a parametrizar. Al momento: 'MLEM Parameter', 'GenerateImage Parameter'
 * - el máximo largo por línea es de 512 caracteres.
 * - una línea tiene dos strins separados por ':=' (keyword := valor) y con un espacio a cada lado del ':='.
 * - el máximo largo de cada keyword y valor es de 256 caracteres.
 * - se da por finalizado el archivo con la línea 'END:'
 * - una línea iniciando con punto y coma ';' es un comentario
 * - Existen subgrupos de parámetros que por ahora no son tratados como subgrupos, por lo que los espacios
 *   o '\t' al principio de un campo de un subgrupo son eliminados y se los considera como una entrada común. 
 *
 * @param fileName Nombre del archivo de parámetros.
 * @param parameterType Nombre del tipo de parámetros a definir (Nombre del comando, por lo general).
 * @param searchWords array de strings con los keyword a buscar.
 * @param numWords entero con la cantidad de words que se requieren.
 * @param returnValues arrays con los valores para cada keyword. 
 * @param errorMessage mensaje de error o warning. 
 * @return 0 si no uhbo errores, 1  si alguno de los keyword aparece más de una vez en el archivo,
 * 2  si alguno de los keyword no fue encontrado (returned value is empty (i.e. contains '/0's only)),
 * 3  el archivo no pude abrirse para lectura (returned value is empty (i.e. contains '/0's only)),
 * 4  Posiblemente el archivo tiene otro formato (No se encontró el METODOParameters al inicio, o ese método no esta disponible) (returned value is empty (i.e. contains '/0's only)),
 * 5  el método buscado no es válido.
 */
int parametersFile_readMultipleKeys(char* fileName, char* parameterType, char** searchWords, int numWords, char** returnValue, char* errorMessage)
{
  int i = 0, readResult = 0;
  for(i = 0; i < numWords; i++)
  {
    if((readResult = parametersFile_read(fileName, parameterType, searchWords[i], returnValue[i], errorMessage)) != 0)
    {
      // Hubo una keyword que falló, salgo de la función y de vuelvo el código de error de esa keyword.
      return readResult;
    }
  }
  return PMF_NO_ERROR;
}

/*!
 * La función busca múltiples palabras claves de campos del archivo de parámetros y 
 * devuelve los valores correspondiente a cada campo.
 * Para esto es necesario pasarle como parámetros el nombre del archivo de parámetros,
 * el tipo de parámetros del archivo (depende del comando, por lo gral se lo identifica 
 * directamente con el nombre del comando), el nombre del sector
 * y los nombres de los campos que se desea buscar. Los valores de los campos
 * son devueltos en un array de strings, o sea un doble puntero a char*.
 * La interpretación del campo deberá hacerse desde el programa que se llama
 * a esta función.
 *
 * Un sector dentro de un archivo de parámetros se caracteriza por tener como línea
 * identificatoria de la sección el nombrel del grupo de parámetros. Luego dentro
 * del mismo tiene todos los subparámetros correspondientes a ese sector y finaliza
 * con el nombre del sector precedido por la palabra End. Ejemplo:
 * \code
	output file format type := Interfile
	interfile Output File Format Parameters :=
		number format  := unsigned integer
		number_of_bytes_per_pixel := 2
		; fix the scale factor to 1
		; comment out next line to let STIR use the full dynamic 
		; range of the output type
		scale_to_write_data := 1
	End Interfile Output File Format Parameters :=
 * \endcode
 *
 * @param fileName Nombre del archivo de parámetros.
 * @param parameterType Nombre del tipo de parámetros a definir (Nombre del comando, por lo general).
 * @param searchSector string con el nombre del secto a buscar.
 * @param searchWords array de strings con los keyword a buscar.
 * @param returnValues arrays con los valores para cada keyword. 
 * @param errorMessage mensaje de error o warning. 
 * @return 0 si no uhbo errores, 1  si alguno de los keyword aparece más de una vez en el archivo,
 * 2  si alguno de los keyword no fue encontrado (returned value is empty (i.e. contains '/0's only)),
 * 3  el archivo no pude abrirse para lectura (returned value is empty (i.e. contains '/0's only)),
 * 4  Posiblemente el archivo tiene otro formato (No se encontró el METODOParameters al inicio, o ese método no esta disponible) (returned value is empty (i.e. contains '/0's only)),
 * 5  el método buscado no es válido.
 */
int parametersFile_readSector(char* fileName, char* parameterType, char* searchSector, char** searchWords, char** returnValues, char* errorMessage)
{
	return 0;
}