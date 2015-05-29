

#include <Sinogram3D.h>


//using namespace::std;
//std::ostringstream os;

//using namespace::iostream;
Sinogram3D::Sinogram3D(float rFov_mm, float zFov_mm)
{
  // Inicializo la geometría del scanner:
  radioFov_mm = rFov_mm;
  axialFov_mm = zFov_mm;
  //Segmentos = (Segmento**) malloc(sizeof(Segmento*));
  numSegments = 0;
}

Sinogram3D::Sinogram3D(char* fileHeaderPath, float rFov_mm, float zFov_mm)
{
  // Inicializo la geometría del scanner:
  radioFov_mm = rFov_mm;
  axialFov_mm = zFov_mm;
  //Segmentos = (Segmento**) malloc(sizeof(Segmento*));
  numSegments = 0;
  // Reemplazo la función que lee el interfile del discovery STE por la genérica:
  if(readFromInterfile(fileHeaderPath))
  {
    return;
  }
  /*if(ReadInterfileDiscoverySTE(fileHeaderPath)==false)
  {
	cout<<strError<<endl;
	return;
  }*/
}

/// Constructor para copia desde otro objeto sinograma3d
Sinogram3D::Sinogram3D(Sinogram3D* srcSinogram3D)
{
  /// Copio todas las propiedaddes del objeto fuente al objeto siendo instanciado.
  this->radioFov_mm = srcSinogram3D->radioFov_mm;
  this->axialFov_mm = srcSinogram3D->axialFov_mm;
  this->numProj = srcSinogram3D->getNumProj();
  this->numR = srcSinogram3D->getNumR();
  this->numSegments = srcSinogram3D->getNumSegments();
  this->numRings = srcSinogram3D->getNumRings();
  this->maxRingDiff = srcSinogram3D->getMaxRingDiff();
  ptrAxialvalues_mm = (float*) malloc(sizeof(float)*numRings);
  for(int i = 0; i < numRings; i++)
  {
    ptrAxialvalues_mm[i] = srcSinogram3D->getAxialValue(i);
  }
  // Configuración de los segmentos:
  numSinogramsPerSegment = (int*) malloc(sizeof(int)*numSegments);
  memcpy(numSinogramsPerSegment, srcSinogram3D->numSinogramsPerSegment, sizeof(int)*numSegments);
  minRingDiffPerSegment = (int*) malloc(sizeof(int)*numSegments);
  memcpy(minRingDiffPerSegment, srcSinogram3D->minRingDiffPerSegment, sizeof(int)*numSegments);
  maxRingDiffPerSegment = (int*) malloc(sizeof(int)*numSegments);
  memcpy(maxRingDiffPerSegment, srcSinogram3D->maxRingDiffPerSegment, sizeof(int)*numSegments);
}


/// Desctructor
Sinogram3D::~Sinogram3D()
{
  free(ptrAxialvalues_mm);
}

int Sinogram3D::getBinCount()
{
  int numBins = 0;
  // Obtengo la cantidad total de bins que tiene el sinograma 3D.
  for(int i = 0; i < this->numSegments; i++)
  {
    for(int j = 0; j < this->getSegment(i)->getNumSinograms(); j++)
    {
      numBins += this->getSegment(i)->getSinogram2D(j)->getNumProj() * this->getSegment(i)->getSinogram2D(j)->getNumR();
    }
  }
  return numBins;
}

int Sinogram3D::getNumSinograms()
{
  int numSinos = 0;
  // Obtengo la cantidad total de sinos 2d que tiene el sinograma 3D.
  for(int i = 0; i < this->numSegments; i++)
  {
    numSinos += this->getSegment(i)->getNumSinograms();
  }
  return numSinos;
}


bool Sinogram3D::readFromInterfile(string headerFilename)
{
  // Set a dummy value for the radio of the scanner and then call the funcion that actully reads the file:
  float radioScanner_mm = radioFov_mm;
  readFromInterfile(headerFilename, radioScanner_mm);
  return true;
}


bool Sinogram3D::readFromInterfile(string headerFilename, float radioScanner_mm)
{
  const int sizeBuffer = 1024;
  char buffer[sizeBuffer];	// Buffer que voy a utilizar para ir leyendo el archivo de configuración
  char errorMessage[sizeBuffer];
  char filename_i33[sizeBuffer];
  char header[] = "INTERFILE";
  int numDims = 0;

  // Por ahora hago una lectura muy simple:
  // Número de dimensiones:
  if (interfile_read((char*)headerFilename.c_str(), "!name of data file", filename_i33, errorMessage)) 
  {
    // Error de lectura:
    cout << "Error al buscar name of data file en lectura de header del Sinograma3D: " << errorMessage << endl;
    return false;
  }
  // Si el filename tiene el path completo lo dejo como está si solo tiene el nombre del archivo le adjunto
  // el path del header:
  char* ultimaBarra = strrchr(filename_i33, PATH_BAR);
  if(ultimaBarra == NULL)
  {
    // Agregp el path del headerFilename
    ultimaBarra = strrchr((char*)headerFilename.c_str(), PATH_BAR);
    // Uso buffer como intermedio. Primero copio el header:
    strncpy(buffer, headerFilename.c_str(), (ultimaBarra-headerFilename.c_str()+1));
    // strncpy no agrega el cero de fin de string:
    buffer[ultimaBarra-headerFilename.c_str()+1] = '\0';
    // Luego concateno:
    strcat(buffer, filename_i33);
    // Finalmente copio todo en filename_i33_
    strcpy(filename_i33, buffer);
  }
  
  // Por ahora hago una lectura muy simple:
  // Número de dimensiones:
  if (interfile_read((char*)headerFilename.c_str(), "number of dimensions", buffer, errorMessage)) 
  {
    // Error de lectura:
    cout << "Error en lectura de Sinograma3D: " << errorMessage << endl;
    return false;
  }
  else
  {
    numDims = atoi(buffer);
    if (numDims != 4)
    {
      // Error de lectura:
      cout << "Error en interfile: Un sinograma 3d debe tener cuatro dimensiones." << endl;
      return false;
    }
  }
  
  // Número de anillos (por ahora lo saco de la cantidad de sinogramas del primer segmento:
  /*if (interfile_read((char*)headerFilename.c_str(), "Number of rings", buffer, errorMessage)) 
  {
    // Error de lectura:
    cout << "Error en lectura de Sinograma3D: " << errorMessage << endl;
    return false;
  }
  else
  {
    numRings = atoi(buffer);
    if (numDims <= 0)
    {
      // Error de lectura:
    cout << "Error en interfile: El scanner tiene cero anillos." << endl;
    return false;
    }
  }*/
  
  // Ahora los datos por cada dimensión, esto se podría hacer más automático en el que cada
  // dimensión puede ser cualquiera de las variables involucradas, pero por ahora lo hago de forma
  // fija, dimensión 4 segmentos, 2 vistas, 3 coordenadas axials(sinos por anillo), 1 muestras transversales.
  if (interfile_read((char*)headerFilename.c_str(), "matrix axis label [4]", buffer, errorMessage)) 
  {
    // Error de lectura:
    cout << "Error en lectura de Sinograma3D: " << errorMessage << endl;return false;
  }
  else
  {
    if(!strcmp(buffer,"segment"))
    {
	if (interfile_read((char*)headerFilename.c_str(), "!matrix size [4]", buffer, errorMessage)) 
	{
	  // Error de lectura:
	  cout << "Error en lectura de Sinograma3D: " << errorMessage << endl;return false;
	}
	else
	{
	  numSegments = atoi(buffer);
	  if (numSegments <= 0)
	  {
	    // Error de lectura:
	    cout << "Error en interfile: El sinograma tiene cero segmentos." << endl;return false;
	  }
	}
    }
    else  {
      cout << "Error en interfile: La dimensión 4 debe ser los segmentos." << endl;return false;}
  }
  
  
  // Ahora la cantidad de sinogramas por segmento, y las mínimas y máximas diferencias entre anillos de cada segmento.
  numSinogramsPerSegment = (int*) malloc(sizeof(int)*numSegments);
  minRingDiffPerSegment = (int*) malloc(sizeof(int)*numSegments);
  maxRingDiffPerSegment = (int*) malloc(sizeof(int)*numSegments);
  // Leo dichos parámetros:
  if (interfile_read((char*)headerFilename.c_str(), "matrix axis label [3]", buffer, errorMessage)) 
  { // Error de lectura:
    cout << "Error en lectura de Sinograma3D: " << errorMessage << endl;return false;}
  else if(!strcmp(buffer,"axial coordinate"))
  {
    if (interfile_read((char*)headerFilename.c_str(), "!matrix size [3]", buffer, errorMessage)) {
      cout << "Error en lectura de Sinograma3D: " << errorMessage << endl;return false;}
    else
    {
      // Ahora debo ir leyendo la cantidad de segmentos, recordamos que están con el formato {32,23,23,.222}.
      // Primero verifico que me quedo con el string que se encuentra entre la { y la }:
      char* aux = strrchr( buffer, '{'); // Este es el inicio del buffer
      char* end = strrchr( buffer, '}'); // Este es el inicio del buffer
      if ((aux == NULL)||(end == NULL)){
		cout << "No se encontraron los {} de la lista de sinogramas por segmento." << endl; return false;}
      // Si llegué hasta acá pongo el fin del string donde estaba la llave de cierre y saltea una posición donde estaba la de apertura:
      aux++;
      (*end) = '\0';
      // Ahora obtengo todos los strings separados por coma, para eso uso la función strtok:
      char* auxElemento = strtok (aux," ,");
      numSinogramsPerSegment[0] = atoi(auxElemento);
      for(int i = 1; i < numSegments; i++)
      {
	auxElemento = strtok (NULL, " ,.-");
	if(auxElemento == NULL){
	  cout << "La lista de sinogramas por segmento tiene menos elementos que el número de segmentos." << endl; return false;}
	numSinogramsPerSegment[i] = atoi(auxElemento);
      }
      // Verifico que no quedaba ningún elemento adicional:
      auxElemento = strtok (NULL, " ,.-");
      if(auxElemento != NULL){
	  cout << "La lista de sinogramas por segmento tiene más elementos que el número de segmentos." << endl; return false;
      }
    }
  }
  else{  
    cout << "Error en interfile: La dimensión 3 debe ser axial coordinate." << endl;return false;}
  
  if (interfile_read((char*)headerFilename.c_str(), "minimum ring difference per segment", buffer, errorMessage)) {
    cout << "Error en lectura de Sinograma3D: " << errorMessage << endl;return false;}
  else
  {
    // Ahora debo ir leyendo la cantidad de segmentos, recordamos que están con el formato {32,23,23,.222}.
    // Primero verifico que me quedo con el string que se encuentra entre la { y la }:
    char* aux = strrchr( buffer, '{'); // Este es el inicio del buffer
    char* end = strrchr( buffer, '}'); // Este es el inicio del buffer
    if ((aux == NULL)||(end == NULL)){
      cout << "No se encontraron los {} de la lista de sinogramas por segmento." << endl; return false;}
    // Si llegué hasta acá pongo el fin del string donde estaba la llave de cierre y saltea una posición donde estaba la de apertura:
    aux++;
    (*end) = '\0';
    // Ahora obtengo todos los strings separados por coma, para eso uso la función strtok:
    char* auxElemento = strtok (aux," ,");
    minRingDiffPerSegment[0] = atoi(auxElemento);
    for(int i = 1; i < numSegments; i++)
    {
      auxElemento = strtok (NULL, " ,");
      if(auxElemento == NULL){
	cout << "La lista de sinogramas por segmento tiene menos elementos que el número de segmentos." << endl; return false;}
      minRingDiffPerSegment[i] = atoi(auxElemento);
    }
    // Verifico que no quedaba ningún elemento adicional:
    auxElemento = strtok (NULL, " ,");
    if(auxElemento != NULL){
	cout << "La lista de sinogramas por segmento tiene más elementos que el número de segmentos." << endl; return false;}
  }
  
  if (interfile_read((char*)headerFilename.c_str(), "maximum ring difference per segment", buffer, errorMessage)) {
    cout << "Error en lectura de Sinograma3D: " << errorMessage << endl;return false;}
  else
  {
    // Ahora debo ir leyendo la cantidad de segmentos, recordamos que están con el formato {32,23,23,.222}.
    // Primero verifico que me quedo con el string que se encuentra entre la { y la }:
    char* aux = strrchr( buffer, '{'); // Este es el inicio del buffer
    char* end = strrchr( buffer, '}'); // Este es el inicio del buffer
    if ((aux == NULL)||(end == NULL)){
      cout << "No se encontraron los {} de la lista de sinogramas por segmento." << endl; return false;}
    // Si llegué hasta acá pongo el fin del string donde estaba la llave de cierre y saltea una posición donde estaba la de apertura:
    aux++;
    (*end) = '\0';
    // Ahora obtengo todos los strings separados por coma, para eso uso la función strtok:
    char* auxElemento = strtok (aux," ,");
    maxRingDiffPerSegment[0] = atoi(auxElemento);
    for(int i = 1; i < numSegments; i++)
    {
      auxElemento = strtok (NULL, " ,");
      if(auxElemento == NULL){
	cout << "La lista de sinogramas por segmento tiene menos elementos que el número de segmentos." << endl; return false;}
      maxRingDiffPerSegment[i] = atoi(auxElemento);
    }
    // Verifico que no quedaba ningún elemento adicional:
    auxElemento = strtok (NULL, " ,");
    if(auxElemento != NULL){
	cout << "La lista de sinogramas por segmento tiene más elementos que el número de segmentos." << endl; return false;}
  }
 
  // Ahora necesito los valores de numAng y numR:
  if (interfile_read((char*)headerFilename.c_str(), "matrix axis label [2]", buffer, errorMessage)) 
  { // Error de lectura:
    cout << "Error en lectura de Sinograma3D: " << errorMessage << endl;return false;}
  else if(!strcmp(buffer,"view"))
  {
    if (interfile_read((char*)headerFilename.c_str(), "!matrix size [2]", buffer, errorMessage)) {
      cout << "Error en lectura de Sinograma3D: " << errorMessage << endl;return false;}
    else
    {
      this->numProj = atoi(buffer);
    }
  }
  else  {
    cout << "Error en interfile: La dimensión 3 debe ser view." << endl;return false;}

  // Ahora necesito los valores de numAng y numR:
  if (interfile_read((char*)headerFilename.c_str(), "matrix axis label [1]", buffer, errorMessage)) 
  { // Error de lectura:
    cout << "Error en lectura de Sinograma3D: " << errorMessage << endl;return false;}
  else if(!strcmp(buffer,"tangential coordinate"))
  {
    if (interfile_read((char*)headerFilename.c_str(), "!matrix size [1]", buffer, errorMessage)) {
      cout << "Error en lectura de Sinograma3D: " << errorMessage << endl;return false;}
    else
    {
      this->numR = atoi(buffer);
    }
  }
  else  {
    cout << "Error en interfile: La dimensión 1 debe ser tangential coordinate." << endl;return false;}
  
  // La cantidad de anillos, la obtengo de una entrada del interfile, y si no esta disponible supongo que la cantidad de rings es (numSinosPrimerSegmento+1)/2
  // como seria apara cuaqlueri valor de span distinto de uno:
  if (!interfile_read((char*)headerFilename.c_str(), "number of rings", buffer, errorMessage)) 
  { 
    // Se encontro el campo intefilwrite:
    numRings = atoi(buffer);
  }
  else
  {
    // No estaba la leyenda por lo que calculo para un segmento distinto de 1:
    numRings = (int)floor((numSinogramsPerSegment[0]+1)/2);
  }
  
  // Con la cantidad de anillos genero las coordenadas de los anillos:
  ptrAxialvalues_mm = (float*) malloc(numRings*sizeof(float));
  float zIncrement = (float)axialFov_mm/numRings;
  for(int i = 0; i < numRings; i ++)
  {
	  // Initialization of Z Values
	  ptrAxialvalues_mm[i] = zIncrement/2 + i * zIncrement;
  }
  
  inicializarSegmentos();
  
  // Ahora voy llenando cada uno de ellos y leyendo el archivo *.i33:
  FILE* fp = NULL;
  fp = fopen(filename_i33, "r");
  if (fp == NULL)
  {
    cout << "No se pudo abrir el archivo con los datos del sinograma3d." << endl;
  }
  int indiceSino = 0;
  // Para cada sinograma de cada segmento, necesito una lista de de la combinación de anillos que representan
  // ya que se fusionan varios sinos del michelograma en uno solo. Genero una lista auxiliar con la cantidad
  // maxima de anillos que puede tener un sinograma, que es ceil(span/2):
  span = abs(maxRingDiffPerSegment[0]) + abs(minRingDiffPerSegment[0])+1;
  int maxRingsPerSino = (int)ceil((float)span/2);
  int* listaRing1, *listaRing2;
  float *listaZ1_mm, *listaZ2_mm;
  listaRing1 = (int*) malloc(sizeof(int)*maxRingsPerSino);
  listaRing2 = (int*) malloc(sizeof(int)*maxRingsPerSino);
  listaZ1_mm = (float*) malloc(sizeof(float)*maxRingsPerSino);
  listaZ2_mm = (float*) malloc(sizeof(float)*maxRingsPerSino);
  // Reservo un bloque de memoria donde voy guardando cada sinograma que leo:
  float* rawSino = (float*) malloc(sizeof(float)*numProj*numR);
  // z1 y z2 van de 0 a numRings-1.
  for(int i = 0; i<numSegments; i++)
  {
    int numSinosThisSegment = 0;
    // Por cada segmento, voy generando los sinogramas correspondientes y
    // contándolos, debería coincidir con los sinogramas para ese segmento:
    for(int z1 = 0; z1 < numRings*2; z1++)
    {
      int numSinosZ1inSegment = 0;	// Cantidad de sinogramas para z1 en este segmento.
      // Recorro completamente z2 desde y me quedo con los que están entre
      // minRingDiff y maxRingDiff. Se podría hacer sin recorrer todo el
      // sinograma pero se complica un poco.
      int z1_aux = z1;
      for(int z2 = 0; z2 < numRings; z2++)
      {
		// Ahora voy avanzando en los sinogramas correspondientes,
		// disminuyendo z1 y aumentnado z2 hasta que la diferencia entre
		// anillos llegue a maxRingDiff.
		if (((z1_aux-z2)<=maxRingDiffPerSegment[i])&&((z1_aux-z2)>=minRingDiffPerSegment[i]))
		{
			// Me asguro que esté dentro del tamaño del michelograma:
			if ((z1_aux>=0)&&(z2>=0)&&(z1_aux<numRings)&&(z2<numRings))
			{
			//Agrego el sinograma a la lista de anillos:
			listaRing1[numSinosZ1inSegment] = z1_aux;
			listaRing2[numSinosZ1inSegment] = z2;
			// También las coordenadas axiales:
			listaZ1_mm[numSinosZ1inSegment] = ptrAxialvalues_mm[z1_aux];
			listaZ2_mm[numSinosZ1inSegment] = ptrAxialvalues_mm[z2];
			// Incremento:
			numSinosZ1inSegment = numSinosZ1inSegment + 1;
	      
			}
		}
		// Pase esta combinación de (z1,z2), paso a la próxima:
		z1_aux = z1_aux - 1;
      }
      if(numSinosZ1inSegment>0)
      {
		// Si entré acá significa que hay combinaciones de anillos para este sinograma. Lo leo del i33 y lo cargo:
		// Leo un sinograma del i33 y lo asigno al sinograma del segmento correspondiente:
		int numBytes = (int)fread(rawSino, sizeof(float), numProj*numR, fp);
		if (numBytes != (numProj*numR))
		{
			cout << "Falló la lectura de los datos del sinograma3d, puede que no coincida la cantidad de bytes a leer con los del archivo .i33" << endl;
			return false;
		}
		float* ptrSino = this->getSegment(i)->getSinogram2D(numSinosThisSegment)->getSinogramPtr();
		this->getSegment(i)->getSinogram2D(numSinosThisSegment)->setMultipleRingConfig(numSinosZ1inSegment, listaRing1, listaRing2, listaZ1_mm, listaZ2_mm);
		memcpy(ptrSino, rawSino, sizeof(float) * numProj * numR);

		// Cuenta la cantidad de segmentos para verificar que se cumpla:
		numSinosThisSegment = numSinosThisSegment + 1;
		indiceSino = indiceSino + 1;
      }
    }
    if(numSinosThisSegment != numSinogramsPerSegment[i])
    {
      cout << "No coincide la cantidad de sinogramas leídos de un segmento con los definidos en el header." << endl;
      return false;
    }
  }
  fclose(fp);
  return true;
}

bool Sinogram3D::FillConstant(float Value)
{
  /// Se llena todos los bins del sinograma con un valor constante de valor Value.
  /// Esto puede ser de utilidad para calcular el sensibility volume.
  for(int i=0; i< numSegments; i++)
  {
    Segment* auxSegment = this->getSegment(i);
    for(int j=0; j < auxSegment->getNumSinograms(); j++)
    {
      for(int k=0; k < auxSegment->getSinogram2D(j)->getNumProj(); k++)
      {
		for(int l=0; l < auxSegment->getSinogram2D(j)->getNumR(); l++)
		{
		  auxSegment->getSinogram2D(j)->setSinogramBin(k,l, Value);
		}
      }
    }
  }
  return true;
}


// Method that reads the Michelogram data from a file. The dimensions of the
// expected Michelogram are the ones loaded in the constructor of the class
bool Sinogram3D::SaveInFile(char* filePath)
{
  FILE* fileSinogram3D = fopen(filePath,"wb");
  unsigned int CantBytes;
  for(int i = 0; i < numSegments; i++)
  {
    Segment* auxSegment = this->getSegment(i);
    for(int j = 0; j < auxSegment->getNumSinograms(); j++)
    {
      float* ptrSinogram2D = auxSegment->getSinogram2D(j)->getSinogramPtr();
      if((CantBytes =  (int)fwrite(ptrSinogram2D, sizeof(float), auxSegment->getSinogram2D(j)->getNumProj()*auxSegment->getSinogram2D(j)->getNumR() , fileSinogram3D)) !=  (auxSegment->getSinogram2D(j)->getNumProj()*auxSegment->getSinogram2D(j)->getNumR()))
	    return false;
    }
    delete auxSegment;
  }
  fclose(fileSinogram3D);
  return true;
}

float Sinogram3D::getLikelihoodValue(Sinogram3D* referenceProjection)
{
  float likelihood = 0;
  for(int i = 0; i < this->numSegments; i++)
  {
    for(int j = 0; j < this->getSegment(i)->getNumSinograms(); j++)
    {
      for(int k = 0; k < this->getSegment(i)->getSinogram2D(j)->getNumProj(); k++)
      {
		for(int l = 0; l < this->getSegment(i)->getSinogram2D(j)->getNumR(); l++)
		{
		  if(this->getSegment(i)->getSinogram2D(j)->getSinogramBin(k,l) != 0)
		  {
			likelihood += referenceProjection->getSegment(i)->getSinogram2D(j)->getSinogramBin(k,l) 
			  * log(this->getSegment(i)->getSinogram2D(j)->getSinogramBin(k,l))
			  - this->getSegment(i)->getSinogram2D(j)->getSinogramBin(k,l);
		  }
		}
      }
    }
  }
  return likelihood;
}


bool Sinogram3D::writeInterfile(string headerFilename)
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
  fileStream << "!type of data := PET" << eol;
  fileStream << "!imagedata byte order := LITTLEENDIAN" << eol;
  fileStream << "!number of energy windows := 1" << eol;
  fileStream << "energy window [1] := F18m" << eol;
  fileStream << "energy window lower level [1] := 430" << eol;
  fileStream << "energy window upper level [1] := 620" << eol;
  fileStream << "flood corrected := N" << eol;
  fileStream << "decay corrected := N" << eol;

  fileStream << "!PET STUDY (General) :=" << eol;
  // Por ahora el único tipo de dato del sinograma es float:
  fileStream << "!number format := short float" << eol;
  fileStream << "!number of bytes per pixel := " << sizeof(float) << eol;
  /* Por ahora no lo pongo al scaling factor, porque si no coincide con el de generación de datos me caga.
  fileStream << "scaling factor (mm/pixel) [1] := " << (this->getRValue(1)-this->getRValue(0)) << eol;
  fileStream << "scaling factor (deg/pixel) [2] := " << (this->getAngValue(1)-this->getAngValue(0)) << eol;
  */
  // Campos propios del Sinogram3D propuestos en STIR:
  fileStream << "number of dimensions := 4" << eol;
  // Segmentos
  fileStream << "matrix axis label [4] := segment" << eol;
  fileStream << "!matrix size [4] := " << this->getNumSegments() << eol;
  // Cantidad de elementos en y (filas) o de ángulos:
  fileStream << "matrix axis label [2] := view" << eol;
  fileStream << "!matrix size [2] := " << this->getSegment(0)->getSinogram2D(0)->getNumProj() << eol;
  // Cantidad de elementos en x (columnas) o se de posiciones R:
  fileStream << "matrix axis label [1] := tangential coordinate" << eol;
  fileStream << "!matrix size [1] := " << this->getSegment(0)->getSinogram2D(0)->getNumR() << eol;
  // Coordenada axial, sinogramas por segmento:
  fileStream << "matrix axis label [3] := axial coordinate" << eol;
  fileStream << "!matrix size [3] := {";
  for(int i = 0; i< this->getNumSegments()-1; i++)
  {
    fileStream << this->getSegment(i)->getNumSinograms() << ",";
  }
  fileStream << this->getSegment(this->getNumSegments()-1)->getNumSinograms() << "}" << eol;
  // Diferencia entre anillos:
  fileStream << "minimum ring difference per segment := {";
  for(int i = 0; i< this->getNumSegments()-1; i++)
  {
    fileStream << this->getSegment(i)->getMinRingDiff() << ",";
  }
  fileStream << this->getSegment(this->getNumSegments()-1)->getMinRingDiff() << "}" << eol;
  fileStream << "maximum ring difference per segment := {";
  for(int i = 0; i< this->getNumSegments()-1; i++)
  {
    fileStream << this->getSegment(i)->getMaxRingDiff() << ",";
  }
  fileStream << this->getSegment(this->getNumSegments()-1)->getMaxRingDiff() << "}" << eol;
  fileStream << "number of rings := " << this->getNumRings() << eol;
  fileStream << "!data offset in bytes[1] := 0" << eol;
  fileStream << "!number of time frames := 1" << eol;
  fileStream << "!extent of rotation := " << (this->maxAng_deg - this->minAng_deg) << eol;
  // Máximo valor de píxels:
  float max = this->getSinogramBin(0,0);
  for(int i = 0; i < this->numSegments; i++)
  {
    for(int j = 0; j < this->getSegment(i)->getNumSinograms(); j++)
    {
      for(int k = 0; k <  this->getSegment(i)->getSinogram2D(j)->getNumProj(); k++)
      {
	for(int l = 0; l < this->getSegment(i)->getSinogram2D(j)->getNumR(); l++)
	{
	  if(this->getSegment(i)->getSinogram2D(j)->getSinogramBin(k,l) > max)
	    max = this->getSegment(i)->getSinogram2D(j)->getSinogramBin(k,l);
	}
      }
    }
  }
  fileStream << "!maximum pixel count := " << max << eol;
  fileStream << "!END OF INTERFILE :=\n" << eol;
  fileStream.close();
  
  // Ya terminé con el header, ahora escribo el archivo binario:
  fileStream.open(dataFilename.c_str(), ios_base::binary);
  for(int i = 0; i < this->numSegments; i++)
  {
	for(int j = 0; j < this->getSegment(i)->getNumSinograms(); j++)
	{
	  fileStream.write((char*)this->getSegment(i)->getSinogram2D(j)->getSinogramPtr(), this->getSegment(i)->getSinogram2D(j)->getNumProj()*this->getSegment(i)->getSinogram2D(j)->getNumR()*sizeof(float));
	}
  }
  fileStream.close();
  return true;
}

bool correctSinogram (string acfSinogram, string delayedSinogram, string scatterSinogram)
{
  return false;
}

void Sinogram3D::divideBinToBin(Sinogram3D* sinogramDivisor)
{
  float numerador, denominador;
  for(int i = 0; i < this->numSegments; i++)
  {
    for(int j = 0; j < this->getSegment(i)->getNumSinograms(); j++)
    {
      for(int k = 0; k <  this->getSegment(i)->getSinogram2D(j)->getNumProj(); k++)
      {
	for(int l = 0; l < this->getSegment(i)->getSinogram2D(j)->getNumR(); l++)
	{
	  numerador = this->getSegment(i)->getSinogram2D(j)->getSinogramBin(k,l);
	  denominador = sinogramDivisor->getSegment(i)->getSinogram2D(j)->getSinogramBin(k,l);
	  if((numerador != 0)&&(denominador!=0))
	  {
	    this->getSegment(i)->getSinogram2D(j)->setSinogramBin(k,l, numerador/denominador);
	  }
	  else
	  {
	    this->getSegment(i)->getSinogram2D(j)->setSinogramBin(k,l, 0);
	  }
	}
      }
    }
  }
}

void Sinogram3D::multiplyBinToBin(Sinogram3D* sinogramFactor)
{
  for(int i = 0; i < this->numSegments; i++)
  {
    for(int j = 0; j < this->getSegment(i)->getNumSinograms(); j++)
    {
      for(int k = 0; k <  this->getSegment(i)->getSinogram2D(j)->getNumProj(); k++)
      {
	for(int l = 0; l < this->getSegment(i)->getSinogram2D(j)->getNumR(); l++)
	{
	  this->getSegment(i)->getSinogram2D(j)->setSinogramBin(k,l, this->getSegment(i)->getSinogram2D(j)->getSinogramBin(k,l)*sinogramFactor->getSegment(i)->getSinogram2D(j)->getSinogramBin(k,l));
	}
      }
    }
  }
}

void Sinogram3D::inverseDivideBinToBin(Sinogram3D* sinogramDividend)
{
  float numerador, denominador;
  for(int i = 0; i < this->numSegments; i++)
  {
    for(int j = 0; j < this->getSegment(i)->getNumSinograms(); j++)
    {
      for(int k = 0; k <  this->getSegment(i)->getSinogram2D(j)->getNumProj(); k++)
      {
	for(int l = 0; l < this->getSegment(i)->getSinogram2D(j)->getNumR(); l++)
	{
	  numerador = sinogramDividend->getSegment(i)->getSinogram2D(j)->getSinogramBin(k,l);
	  denominador = this->getSegment(i)->getSinogram2D(j)->getSinogramBin(k,l);
	  if((numerador != 0)&&(denominador!=0))
	  {
	    this->getSegment(i)->getSinogram2D(j)->setSinogramBin(k,l, numerador/denominador);
	  }
	  else
	  {
	    this->getSegment(i)->getSinogram2D(j)->setSinogramBin(k,l, 0);
	  }
	}
      }
    }
  }
}

void Sinogram3D::addBinToBin(Sinogram3D* sinogramToAdd)
{
  /// Add bin to bin two sinograms.
  for(int i = 0; i < this->numSegments; i++)
  {
    for(int j = 0; j < this->getSegment(i)->getNumSinograms(); j++)
    {
      for(int k = 0; k <  this->getSegment(i)->getSinogram2D(j)->getNumProj(); k++)
      {
	for(int l = 0; l < this->getSegment(i)->getSinogram2D(j)->getNumR(); l++)
	{
	  this->getSegment(i)->getSinogram2D(j)->setSinogramBin(k,l, this->getSegment(i)->getSinogram2D(j)->getSinogramBin(k,l) + sinogramToAdd->getSegment(i)->getSinogram2D(j)->getSinogramBin(k,l));
	}
      }
    }
  }
}
