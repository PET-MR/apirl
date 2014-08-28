

#include <Sinogram3DCylindricalPet.h>


using namespace::std;
std::ostringstream os;

//using namespace::iostream;

Sinogram3DCylindricalPet::Sinogram3DCylindricalPet(char* fileHeaderPath, float rFov_mm, float zFov_mm, float rScanner_mm):Sinogram3D(rFov_mm, zFov_mm)
{
  radioScanner_mm = rScanner_mm;
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
Sinogram3DCylindricalPet::Sinogram3DCylindricalPet(Sinogram3DCylindricalPet* srcSinogram3D):Sinogram3D((Sinogram3D*)srcSinogram3D)
{
  /// Copio todas las propiedaddes del objeto fuente al objeto siendo instanciado en el constructor de Sinogram3d.
  /// Ahora hago la copia de los objetos Segmento
  /// Instancio los sinogramas 2D
  segments = new SegmentInCylindrical3Dpet*[numSegments];
  for(int i = 0; i < numSegments; i++)
  {
    segments[i] = new SegmentInCylindrical3Dpet(srcSinogram3D->getSegment(i));
  }
  // El radio del scnnaer también:
  radioScanner_mm = srcSinogram3D->getRadioScanner_mm();
}

/// Desctructor
Sinogram3DCylindricalPet::~Sinogram3DCylindricalPet()
{
  /// Tengo que eliminar toda la memoria pedida!
  /// Libero los segmentos
  for(int i = 0; i < numSegments; i++)
  {
    delete segments[i];
  }
  free(segments);
  free(ptrAxialvalues_mm);
}

// Constructor que genera un nuevo sinograma 3D a aprtir de los subsets.
Sinogram3DCylindricalPet::Sinogram3DCylindricalPet(Sinogram3DCylindricalPet* srcSinogram3D, int indexSubset, int numSubsets):Sinogram3D((Sinogram3D*)srcSinogram3D)
{
  // Genero un nuevo sinograma 3d que sea un subset del sinograma principal. La cantidad de segmentos, y sinogramas
  // por segmento la mantengo. Lo único que cambia es que cada sinograma 2D, lo reduzco en ángulo numSubsets veces.
  // Para esto me quedo solo con los ángulos equiespaciados en numSubsets partiendo desde indexSubsets.
  /// Copio todas las propiedaddes del objeto fuente al objeto siendo instanciado.
  /// Ahora hago la copia de los objetos Segmento
  inicializarSegmentos();
  
  // Hasta ahora tengo una copia del sinograma 3d anterior, cambio los sinogramas 2d de cada segmento con los del subset.
  for(int i = 0; i < numSegments; i++)
  {
      for(int j = 0; j < srcSinogram3D->getSegment(i)->getNumSinograms(); j++)
      {
	Sinogram2DinCylindrical3Dpet* sino2dCompleto = srcSinogram3D->getSegment(i)->getSinogram2D(j);
	segments[i]->setSinogram2D(new Sinogram2DinCylindrical3Dpet(sino2dCompleto, indexSubset, numSubsets), j);
      }
  }
}

Sinogram3D* Sinogram3DCylindricalPet::Copy()
{
  Sinogram3DCylindricalPet* sino3dcopy = new Sinogram3DCylindricalPet(this);
  return (Sinogram3D*)sino3dcopy;
}

Sinogram3D* Sinogram3DCylindricalPet::getSubset(int indexSubset, int numSubsets)
{
  Sinogram3DCylindricalPet* sino3dSubset = new Sinogram3DCylindricalPet(this);
  // Tengo una copia del sinograma, debo cambiar los sinogramas
  for(int i = 0; i < numSegments; i++)
  {
      for(int j = 0; j < this->segments[i]->getNumSinograms(); j++)
      {
	sino3dSubset->getSegment(i)->setSinogram2D(new Sinogram2DinCylindrical3Dpet(this->getSegment(i)->getSinogram2D(j), indexSubset, numSubsets), j);
      }
  }
  return (Sinogram3D*)sino3dSubset;
}

/// Función que inicializa los segmentos.
void Sinogram3DCylindricalPet::inicializarSegmentos()
{
  /// Instancio los sinogramas 2D
  segments = new SegmentInCylindrical3Dpet*[numSegments];
  for(int i = 0; i < numSegments; i++)
  {
    segments[i] = new SegmentInCylindrical3Dpet(numProj, numR, numRings, radioFov_mm, axialFov_mm, radioScanner_mm, 
	  numSinogramsPerSegment[i], minRingDiffPerSegment[i], maxRingDiffPerSegment[i]);
  }
}
	
/// Setes un sinograma en un segmento.
void Sinogram3DCylindricalPet::setSinogramInSegment(int indexSegment, int indexSino, Sinogram2DinCylindrical3Dpet* sino2d)
{
  segments[indexSegment]->setSinogram2D(sino2d, indexSino);
}
	
	
bool Sinogram3DCylindricalPet::ReadInterfileDiscoverySTE(char* fileDataPath)
{
  /// Inicializo todos los segmentos y las variables de Ring Differences
  const int ConstCantSegmentos = 23;
  numSegments = ConstCantSegmentos;
  /// Primero los vectores con las diferencias entre anillos
  int MaxRingDiffLocal[ConstCantSegmentos] = { 1, 3, -2, 5, -4, 7, -6, 9, -8, 11, -10, 13, -12, 15, -14, 17, -16, 19, -18, 21, -20, 23, -22  };
  int MinRingDiffLocal[ConstCantSegmentos] = { -1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12, -13, 14, -15, 16, -17, 18, -19, 20, -21, 22, -23 };
  /// Tamaño de los Sinogramas
  // Ge:
  numProj = 280;
  numR = 329;
  // Prueba mia:
  numProj = 160;
  numR = 160;
  numRings = 24;
  radioFov_mm = 350;
  /// Radio del Scanner en mm
  radioScanner_mm = 886.2/2;
  axialFov_mm = 156.96;
  /// Ahora los segmentos
  int SinogramasPorSegmento[ConstCantSegmentos] = { 47, 43, 43, 39, 39, 35, 35, 31, 31, 27, 27, 23, 23, 19, 19, 15, 15, 11, 11, 7, 7, 3, 3 };
  ///(Segmentos**) malloc (sizeof(Segmento*));
  /// Reservo memoria para el array de segmentos
  segments = new SegmentInCylindrical3Dpet*[ConstCantSegmentos];
  /// Ahora instancio cada segmento
  for(int i = 0; i < numSegments; i++)
  {
    segments[i] = new SegmentInCylindrical3Dpet(numProj, numR, numRings, radioFov_mm, axialFov_mm, radioScanner_mm, 
						SinogramasPorSegmento[i], MinRingDiffLocal[i], MaxRingDiffLocal[i]);
  }
  FILE* fileSino3D;
  /// Ahora leo los datos binarios y cargo los archivos
  if((fileSino3D=fopen(fileDataPath, "rb")) == NULL)
  {
	strError.assign("Error al abrir el archivo binario que contiene el Sinograma 3D.");
	return false;
  }
  /// Una vez abierto el archivo voy leyendo los sinogramas2D pertenecientes a cada segmento, y
  /// los voy cargando en el array de segmentos.
  int CantBytesPorSino = numR * numProj * sizeof(float);
  float* Sino2D = (float*) malloc(sizeof(float)*CantBytesPorSino);	/// Puntero que define un vector auxiliar para la lectura secuencial de sinogramas 2D
  for(int i = 0; i < numSegments; i++)
  {
	/// Los segmentos deber�an tener (numRings-abs(MinRingDiff))*2 - 1 sinogramas.
	/// Excepto el segmento cero, que al estar centrado, en los casos que no entre
	/// el oblicuo generado con MinRingDiff o MaxRingDiff, siempre se van a poder poner
	/// el directo, por lo que va a tener 2 * numRings -1 sinogramas.
	if( i == 0)
	{
	  /// Segmento central
	  /// Verifico que la cantidad de sinogramas sea la esperada.
	  if (segments[i]->getNumSinograms() != (numRings*2-1))
	  {
		os.clear();
		os << "La cantidad de Sinogramas del Segmento "<< i << " no coincide con la cantidad esperada." << endl;
		strError.assign(os.str());
		return false;
	  }
	  /// Tambien se espera que el central tenga el mismo valor absoluto MinRingDiff que MaxRingdiff
	  /// pero distinto signo.
	  if (segments[i]->getMaxRingDiff() != -segments[i]->getMinRingDiff())
	  {
		os.clear();
		os << "En el segmento central " << i << " se espera que la máxima diferencia entre anillos sea igual en valor absoluto que la mínima." << endl;
		strError.assign(os.str());
		return false;
	  }
	  /// Ahora recorro todos los sinogramas y les asigno las coordenadas de rings de las distintas LORs.
	  /// Se recorre desde el sinograma con posici�n axial m�s cercana al ring1, en direcci�n del �ltimo
	  /// ring. Siempre el primero contiene una cantidad impar de LORs, el segundo para, y va alternando
	  /// entre esos dos valores hasta el final.
	  /// Recorro todos los sinogramas del segmento.
	  int CantLORSimpar;
	  int CantLORSpar;
	  if(!((segments[i]->getMaxRingDiff())%2))
	  {
		/// Si la diferncia entre m�ximo y m�nimo es par, significa que la m�xima
		/// cantidad de LORs por sinograma Segmentos[i]->MaxRingDiff - Segmentos[i]->MinRingDiff + 1
		/// se va dar en los �ndices j pares( o sea impares si se arranca desde 1)
		CantLORSpar = segments[i]->getMaxRingDiff() + 1;
		CantLORSimpar = segments[i]->getMaxRingDiff();
	  }
	  else
	  {
		/// En el caso contrario, la m�xima se da en los �ndices impares( pares sis e arranca desde 1)
		CantLORSpar = segments[i]->getMaxRingDiff();
		CantLORSimpar = segments[i]->getMaxRingDiff() + 1;
	  }
	  /// Los rings van desde 0 a numRings -1
	  int iRing1 = floorf(segments[i]->getMinRingDiff()/2);	/// Empiezo desde un "ring imaginario" negativo, para poder recorrer todo				
	  /// Para el segmento central, supongo que el minringdiff es negativo y el maxringdiff es positivo, e iguales
	  /// en valor absoluto.
	  for(int j = 0; j < segments[i]->getNumSinograms(); j++)
	  {
		float* ptrSinogram2D = segments[i]->getSinogram2D(j)->getSinogramPtr();
		/// Copio Datos del Sinograma
		int n;
		if(( n = fread(ptrSinogram2D, sizeof(float), numR*numProj, fileSino3D))!= (numR*numProj))
			printf("Error al leer los datos del sinograma\n");
		/// En dos vectores auxiliares reservo para la lista de posicionZ la máxima cantidad de LORs posibles para 
		/// el segemtno correspondiente. Luego las lleno, y seteo dicha configuración en el sinograma2d según
		/// la cantidad de combinaciones en Z definidas.
		int numZ = 0;
		int* listaRing1 = (int*)malloc(sizeof(int)*(segments[i]->getMaxRingDiff() + 1));
		int* listaRing2 = (int*)malloc(sizeof(int)*(segments[i]->getMaxRingDiff() + 1));
		float *listaZ1_mm = (float*)malloc(sizeof(float)*(segments[i]->getMaxRingDiff() + 1));
		float *listaZ2_mm = (float*)malloc(sizeof(float)*(segments[i]->getMaxRingDiff() + 1));
		///Siempre el primer sinograma va a tener la cantidad impar de sinogramas.
		if((j%2)==0)
		{
		  // Si el indice es par(impar si se arranca desde 1 en vez de cero) tengo CantLORSimpar = Segmentos[i]->MaxRingDiff - Segmentos[i]->MinRingDiff
		  int indiceAux = 0;
		  for(int k=0; k < CantLORSpar; k++)
		  {
			/// Siempre en el recorrido las posiciones par tienen cantidad impar de LORs, por lo que van centradas
			/// en iRing1.
			/// Cargo todas las LORs. Arrancado por la de extremo, o sea que Ring2 y Ring1 tenga valores entre 0 y Nrings-1
			if(((iRing1 + k)>=0)&&((iRing1 + (CantLORSpar-1) - k)<numRings)&&((iRing1 + k)>=0)&&((iRing1 + (CantLORSpar-1) - k)<numRings))
			{
			  /// Los dos anillos de la LOR caen dentro del scanner, entonces los agrego.
			  /// iRing1 es el extremo izquierdo, no el central!
			  numZ++;
			  listaRing1[indiceAux] = iRing1 + k;
			  listaRing2[indiceAux] = iRing1 + (CantLORSpar-1) - k;
			  listaZ1_mm[indiceAux] = ptrAxialvalues_mm[listaRing1[indiceAux]];
			  listaZ2_mm[indiceAux] = ptrAxialvalues_mm[listaRing2[indiceAux]];
			  indiceAux++;
			}
		  }
		}
		else
		{
		  // Si el indice es impar(par si se arranca desde 1 en vez de cero) tengo CantLORSpar = Segmentos[i]->MaxRingDiff - Segmentos[i]->MinRingDiff + 1
		  int indiceAux = 0;
		  for(int k=0; k < CantLORSimpar; k++)
		  {
			/// Cargo todas las LORs. Arrancado por la de extremo
			if(((iRing1 + k)>=0)&&((iRing1 + (CantLORSimpar-1) - k)>=0)&&((iRing1 + k)<numRings)&&((iRing1 + (CantLORSimpar-1) - k)<numRings))
			{
			  /// Los dos anillos de la LOR caen dentro del scanner, entonces los agrego.
			  /// En este caso la cantidad de lors es par por lo que r1 va desde iring1
			  numZ++;
			  listaRing1[indiceAux] = iRing1 + k;
			  listaRing2[indiceAux] = iRing1 + (CantLORSimpar-1) - k;
			  listaZ1_mm[indiceAux] = ptrAxialvalues_mm[listaRing1[indiceAux]];
			  listaZ2_mm[indiceAux] = ptrAxialvalues_mm[listaRing2[indiceAux]];
			  indiceAux++;
			}
		  }
		  /// Incremento el ring solo 
		  iRing1++;
		}
		/// Seteo la configuración definida:
		segments[i]->getSinogram2D(j)->setMultipleRingConfig(numZ, listaRing1, listaRing2, listaZ1_mm, listaZ2_mm);
	  }
	}
	/// Controlar c�digo a aprtir de ac�!!!!!!!!!! 11/09/09
	else
	{
	  /// El resto de los segmentos.
	  /// Verifico que la cantidad de sinogramas sea la esperada.
	  int MinAbsRing = min(abs(segments[i]->getMinRingDiff()),abs(segments[i]->getMaxRingDiff()));
	  int MaxAbsRing = max(abs(segments[i]->getMinRingDiff()),abs(segments[i]->getMaxRingDiff()));
	  if (segments[i]->getNumSinograms() != ((numRings - MinAbsRing)*2-1))
	  {
		os.clear();
		os << "La cantidad de Sinogramas del Segmento " << i << " no coincide con la cantidad esperada." << endl;
		strError.assign(os.str());
		return false;
	  }
	  /// Ahora recorro todos los sinogramas y les asigno las coordenadas de rings de las distintas LORs.
	  /// Se recorre desde el sinograma con posici�n axial m�s cercana al ring1, en direcci�n del �ltimo
	  /// ring. Siempre el primero contiene una cantidad impar de LORs, el segundo para, y va alternando
	  /// entre esos dos valores hasta el final.
	  /// Recorro todos los singoramas del segmento.
	  int CantLORSimpar = ceilf(((float)(MaxAbsRing - MinAbsRing))/2);
	  int CantLORSpar = ceilf(((float)(MaxAbsRing - MinAbsRing + 1))/2);
	  int iRing1 = - floorf((MaxAbsRing-MinAbsRing)/2);
	  /* Si la diferencia entre anillos es positiva o negativa, lo analizo reci�n a la hora de cargar los anillos
	  en el objeto Sinogram2D porque el c�lculo es totalmente sim�trico. Entonces si la diferencia de anillos es
	  positiva le asigno al Ring1 el indice calculado normalmente, si la diferencia es negativa invierto los valores
	  de ring 1 y ring2
	  if(Segmentos[i]->MinRingDiff>=0)
	  {
		  /// Diferencias positivas entre anillos
		  iRing1 = 1 - floor((Segmentos[i]->MaxRingDiff-Segmentos[i]->MinRingDiff)/2);
		  iRing2 =  1 + Segmentos[i]->MinRingDiff;
	  }
	  else
	  {
		  /// Diferencias negativas entre anillos
		  iRing2 = 1;
		  iRing1 =  1 - Segmentos[i]->MinRingDiff;
	  }
	  iRing1 = floor(Segmentos[i]->MinRingDiff/2);*/
	  for(int j = 0; j < segments[i]->getNumSinograms(); j++)
	  {
		float* ptrSinogram2D = segments[i]->getSinogram2D(j)->getSinogramPtr();
		/// Copio Datos del Sinograma
		if(fread(ptrSinogram2D, sizeof(float), numR*numProj, fileSino3D)!= (numR*numProj))
		  printf("Error al leer los datos del sinograma\n");
		/// En dos vectores auxiliares reservo para la lista de posicionZ la máxima cantidad de LORs posibles para 
		/// el segemtno correspondiente. Luego las lleno, y seteo dicha configuración en el sinograma2d según
		/// la cantidad de combinaciones en Z definidas.
		int numZ = 0;
		int* listaRing1 = (int*)malloc(sizeof(int)*(MaxAbsRing-MinAbsRing+1));
		int* listaRing2 = (int*)malloc(sizeof(int)*(MaxAbsRing-MinAbsRing+1));
		float* listaZ1_mm = (float*) malloc(sizeof(float)*(MaxAbsRing-MinAbsRing+1));
		float* listaZ2_mm = (float*) malloc(sizeof(float)*(MaxAbsRing-MinAbsRing+1));
		/// Arranco con la cantidades impares de LORS, por lo que cuando el �ndice
		/// j sea par, la cantidad de LORs te�ricas va a ser (MaxRingDiff - MinRingDiff) cuando
		/// (MaxRingDiff - MinRingDiff) es impar y (MaxRingDiff - MinRingDiff)
		if((j%2)==0)
		{
		  // Si el indice es par(impar si se arranca desde 1 en vez de cero) tengo CantLORSimpar = Segmentos[i]->MaxRingDiff - Segmentos[i]->MinRingDiff
		  int indiceAux = 0;
		  for(int k=0; k < CantLORSimpar; k++)
		  {
			/// Cargo todas las LORs. Arrancado por la de extremo
			if(((iRing1 + k)>=0)&&((iRing1 + k)<numRings)&&((iRing1 + MaxAbsRing - 1 -k)>=0)&&((iRing1 + MaxAbsRing - k -1)<numRings))
			{
			  /// Los dos anillos de la LOR caen dentro del scanner, entonces los agrego.
			  numZ++;
			  if(segments[i]->getMaxRingDiff()>=0)
			  {
				  listaRing1[indiceAux] = iRing1 + k;
				  listaRing2[indiceAux] = iRing1 + MaxAbsRing -1 - k;
				  listaZ1_mm[indiceAux] = ptrAxialvalues_mm[listaRing1[indiceAux]];
				  listaZ2_mm[indiceAux] = ptrAxialvalues_mm[listaRing2[indiceAux]];
			  }
			  else
			  {
				  /// Diferencia entre anillos negativa, intercambio los valores.
				  listaRing1[indiceAux] = iRing1 + MaxAbsRing -1 - k;
				  listaRing2[indiceAux] = iRing1 + k;
				  listaZ1_mm[indiceAux] = ptrAxialvalues_mm[listaRing1[indiceAux]];
				  listaZ2_mm[indiceAux] = ptrAxialvalues_mm[listaRing2[indiceAux]];
			  }
			  indiceAux++;
			}
		  }
		}
		else
		{
		  // Si el indice es impar(par si se arranca desde 1 en vez de cero) tengo CantLORSpar = Segmentos[i]->MaxRingDiff - Segmentos[i]->MinRingDiff + 1
		  int indiceAux = 0;
		  for(int k=0; k < CantLORSpar; k++)
		  {
			/// Cargo todas las LORs. Arrancado por la de extremo
			if(((iRing1 + k)>=0)&&((iRing1 + k)<numRings)&&((iRing1 + MaxAbsRing - k)>=0)&&((iRing1 + MaxAbsRing - k)<numRings))
			{
			  /// Los dos anillos de la LOR caen dentro del scanner, entonces los agrego.
			  numZ++;
			  if(segments[i]->getMinRingDiff()>=0)
			  {
				listaRing1[indiceAux] = iRing1 + k;
				listaRing2[indiceAux] = iRing1 + MaxAbsRing - k;
				listaZ1_mm[indiceAux] = ptrAxialvalues_mm[listaRing1[indiceAux]];
				listaZ2_mm[indiceAux] = ptrAxialvalues_mm[listaRing2[indiceAux]];
			  }
			  else
			  {
				listaRing1[indiceAux] = iRing1 + MaxAbsRing - k;
				listaRing2[indiceAux] = iRing1 + k;
			  }
			  indiceAux++;
			}
		  }
		  // Solo incrmento el ring en los pares.
		  iRing1++;
		}
		/// Seteo la configuración definida:
		segments[i]->getSinogram2D(j)->setMultipleRingConfig(numZ, listaRing1, listaRing2, listaZ1_mm, listaZ2_mm);
	  }
	}
  }
  fclose(fileSino3D);
  return true;
}

