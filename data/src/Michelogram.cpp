#include <math.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <Utilities.h>
#include <string>
#include <Geometry.h>
#include "Michelogram.h"

using namespace::std;
	
Michelogram::Michelogram(unsigned int myNProj, unsigned int myNR, unsigned int myNZ, unsigned int mySpan, unsigned int myMaxRingDiff, float myRFOV, float myZFOV)
{
	NProj = myNProj;
	NR = myNR;
	NZ = myNZ;
	Span = mySpan;
	MaxRingDiff = myMaxRingDiff;
	/// Radio del Scanner en mm: Ver como hacer para que dependa del equipo
	rScanner = 886.2f/2.0f;
	RFOV = myRFOV;
	ZFOV = myZFOV;
	Sinograms2D = new Sinogram2DinCylindrical3Dpet *[NZ*NZ];	// Allocate memroy for the array of pointers to Sinograms
	// Initialization of each Sinogram object
	for(int i = 0; i < NZ*NZ; i++)
	{
		Sinograms2D[i] = (Sinogram2DinCylindrical3Dpet*)new Sinogram2DinCylindrical3Dpet(NProj, NR, RFOV, rScanner);
		Sinograms2D[i]->setRing1(i%NZ); //Ring 1 : Las columnas
		Sinograms2D[i]->setRing2((unsigned int)(i/NZ));	// Ring 2 las filas
	}
	// Initialization of Z values
	ZValues = (float*) malloc(NZ*sizeof(float));
	ZBins = 2 * NZ - 1;
	float ZIncrement = (float)ZFOV/NZ;
	/// Corregir esto, la diferencia entre anillos debe ser un par�metro!
	DistanceBetweenBins = (ZFOV - ZIncrement/2)/ZBins;
	for(int i = 0; i < NZ; i ++)
	{
		// Initialization of Z Values
		ZValues[i] = ZIncrement/2 + i * ZIncrement;
	}
}

Michelogram::Michelogram(SizeMichelogram MySizeMichelogram)
{
	NProj = MySizeMichelogram.NProj;
	NR = MySizeMichelogram.NR;
	NZ = MySizeMichelogram.NZ;
	Span = 1;
	MaxRingDiff = 1;
	RFOV = MySizeMichelogram.RFOV;
	ZFOV = MySizeMichelogram.ZFOV;
	/// Radio del Scanner en mm: Ver como hacer para que dependa del equipo
	rScanner = 886.2f/2.0f;
	Sinograms2D = new Sinogram2DinCylindrical3Dpet *[NZ*NZ];	// Allocate memroy for the array of pointers to Sinograms
	// Initialization of each Sinogram object
	for(int i = 0; i < NZ*NZ; i++)
	{
		Sinograms2D[i] = (Sinogram2DinCylindrical3Dpet*) new Sinogram2DinCylindrical3Dpet(NProj, NR, RFOV, rScanner);
		Sinograms2D[i]->setRing1(i%NZ); //Ring 1 : Las columnas
		Sinograms2D[i]->setRing2((unsigned int)(i/NZ));	// Ring 2 las filas
	}
	// Initialization of Z values
	ZValues = (float*) malloc(NZ*sizeof(float));
	float ZIncrement = (float)ZFOV/NZ;
	for(int i = 0; i < NZ; i ++)
	{
		// Initialization of Z Values
		ZValues[i] = ZIncrement/2 + i * ZIncrement;
	}
}

Michelogram::~Michelogram()
{
	for(int i = 0; i < NZ*NZ; i++)
	{
		delete Sinograms2D[i];
	}
	delete Sinograms2D;
	free(ZValues);
}
/* Por ahora lo saco, habría que modificarlo :
// Fills the michelograms with a List of Event3D objects received as a parameter
bool Michelogram::Fill(Event3D* Events, unsigned int NEvents)
{
	for(unsigned int i = 0; i < NEvents; i++)
	{
		int BinPhi, BinR, BinZ1, BinZ2;
		float m = (Events[i].Y2 - Events[i].Y1)/(Events[i].X2 - Events[i].X1);
		float Phi = atan((Events[i].Y2 - Events[i].Y1)/(Events[i].X2 - Events[i].X1)) * RAD_TO_DEG + PI_OVER_2;
		// For a LOR defined as y = mx + b, with m = tan(-1/Phi) -> R = b*cos(-1/Phi)
		float b = Events[i].Y1 - (Events[i].X1 * m);
		float R = b * cos(-1/Phi);
		// Now I already have (Phi,R) coordinates fo the LOR. I need to
		// find in which bin of the sinogram this LOR fits.
		if(((BinPhi = SearchBin(Sinograms2D[0]->PhiValues,NProj,Phi))!=-1)&&((BinR = SearchBin(Sinograms2D[0]->RValues,NR,R))!=-1)&&((BinZ1 = SearchBin(ZValues,NZ,Events[i].Z1))!=-1)&&((BinZ2 = SearchBin(ZValues,NZ,Events[i].Z2))!=-1))
		{
			Sinograms2D[BinZ1*NZ+BinZ2]->Sinogram[BinPhi*NR+BinR]++;
			Sinograms2D[BinZ1*NZ+BinZ2]->Ring1 = BinZ1;
			Sinograms2D[BinZ1*NZ+BinZ2]->Ring2 = BinZ2;
		}
	}
	return true;
}
*/

// Function that returns the raw data of the Michelogram
float* Michelogram::RawData()
{
	float* Raw;
	float* ptrSinogram2D;
	Raw = (float*) malloc(sizeof(float)*NZ*NZ*NR*NProj);
	for(int k = 0; k < NZ * NZ; k++)
	{
	  ptrSinogram2D = Sinograms2D[k]->getSinogramPtr();
	  for(int j = 0; j < NProj; j++)
	  {
		  for(int i = 0; i < NR; i++)
		  {
			  Raw[k * (NR * NProj) + j * NR + i] = ptrSinogram2D[j * NR + i];
		  }
	  }
	}
	return Raw;
}

// Function that returns the raw data of the Michelogram
bool Michelogram::FromRawData(float* Raw)
{
  float* ptrSinogram2D;
  for(int k = 0; k < NZ * NZ; k++)
  {
	ptrSinogram2D = Sinograms2D[k]->getSinogramPtr();
	  for(int j = 0; j < NProj; j++)
	  {
		  for(int i = 0; i < NR; i++)
		  {
			  ptrSinogram2D[j * NR + i] = Raw[k * (NR * NProj) + j * NR + i];
		  }
	  }
  }
	return true;
}

// Method that reads the Michelogram data from a file. The dimensions of the
// expected Michelogram are the ones loaded in the constructor of the class
bool Michelogram::readFromFile(string filePath)
{
	FILE* fileMichelogram = fopen(filePath.c_str(),"rb");
	unsigned int CantBytes;
	const unsigned int SizeData = NProj * NR * NZ * NZ;
	float* MichelogramAux = (float*) malloc(SizeData*sizeof(float));
	float* ptrSinogram2D;
	if((CantBytes =  (int)fread(MichelogramAux,SIZE_ELEMENT, SizeData , fileMichelogram)) != SizeData)
		return false;
	// Now I fill the Michelogram
	for(int k = 0; k < NZ * NZ; k++)
	{
	  ptrSinogram2D = Sinograms2D[k]->getSinogramPtr();
		for(int j = 0; j < NProj; j++)
		{
			for(int i = 0; i < NR; i++)
			{
				ptrSinogram2D[j * NR + i] = MichelogramAux[k * (NR * NProj) + j * NR + i];
			}
		}
	}
	fclose(fileMichelogram);
	return true;
}

// Method that reads the Michelogram data from a file. The dimensions of the
// expected Michelogram are the ones loaded in the constructor of the class
bool Michelogram::SaveInFile(char* filePath)
{
	FILE* fileMichelogram = fopen(filePath,"wb");
	unsigned int CantBytes;
	const unsigned int SizeData = NProj * NR * NZ * NZ;
	for(int i = 0; i < NZ * NZ; i++)
	{
		if((CantBytes = (int)fwrite(Sinograms2D[i]->getSinogramPtr(), sizeof(float), NProj*NR , fileMichelogram)) !=  (NProj*NR))
			return false;
	}
	fclose(fileMichelogram);
	return true;
}

// This function reads data from a List Mode file generated with
// a PET scanner Siemens Biograph16, and loads it in Michelogram object
bool Michelogram::ReadDataFromSiemensBiograph16(char* FileName)
{
	// We open the file
	FILE* fid;
	if((fid = fopen(FileName, "rb")) == NULL)
	{
		// Error opening the file
		printf("Error opening the file./n");
		return false;
	}
	// Now we seek for the end of file, so we can 
	// now the size of it
	fseek (fid , 0 , SEEK_END);
	unsigned int Length = ftell (fid);
	// Volvemos el puntero a la posicion inicial
	rewind (fid);

	//Ahora leemos todo
	unsigned int* DatosLista;
	unsigned int Length_in_Int = Length / sizeof(unsigned int);
	DatosLista = (unsigned int*) malloc (sizeof(unsigned int) * Length_in_Int);
	//Leo los datos
	if(fread(DatosLista, sizeof(unsigned int), Length_in_Int, fid) != Length_in_Int)
	{
		printf("Error reading the file./n");
		return false;
	}
	//Procesamos los datos
	//the most significant bit in all event words is the bit 0
	// for further information about the bit sampling, please refer to the Petlink document
	int counter=0;
	int nbins = 192*192*175;
	float* MichelogramBio16 = (float*) malloc(sizeof(float)*nbins);
	for(int i=0; i<nbins; i++)
	   MichelogramBio16[i] = 0;
	int aux;
	for(unsigned int i=0; i<Length_in_Int; i++)
	{    
	   if (!(DatosLista[i] & 0x80000000))
	   { // Si la palabra tiene en el bit 31 un 0 es un evento! 
			aux = DatosLista[i] & 0x1FFFFFF; //El bin addres va desde ocupa los bits b0-b28
			if (aux < nbins)
			{
				// Si el bin esta dentro de los valores esperados
				// lo sumo a su bin correspondiente
				MichelogramBio16[aux]++;
			}
	   }
	}
	// Ahora escribo estos sinogramas en un archivo
	FILE* fidWrite;
	fidWrite = fopen ( "Michelogram_Bio16_float_192_192_175.dat" , "wb" );
	fwrite (MichelogramBio16, sizeof(float) , nbins, fidWrite);
	fclose (fidWrite);

	// Ya tengo la cantidad de eventos por bin del Michelograma.
	// Pero ese Michelograma utiliza Span, y sirve solo para este scanner
	// Entonces lo paso al Michelograma Generico que arme. Para eso a cada
	// bin 
	int indexPhi;
	int indexR;
	int indexZ;
	const int bins_sino2D = 192 * 192;
	int auxr1 = 0;
	int auxr2 = 0;
	for(int i=0; i < nbins; i++)
	{
		indexZ = (int)floorf((float)i/(float)bins_sino2D); // Z its the index of sinograms2D
		// A partir del indice de bin, saco el indice Phi, el indice R, y el indice Z
		indexR = i % 192;	// R are the cloumns
		indexPhi= (int)floorf((i - indexZ * bins_sino2D)/192.0f); // Phi Rows
		// A partir del indice de Z, busco que tipo de Sinograma es.\
		// Y teniendo en cuenta eso sumo los eventos correspondientes
		// all sinogram which include just 1 ring
		if (indexZ==0 || (indexZ>=46 && indexZ<=48) || (indexZ>=84 && indexZ<=87) || 
			(indexZ>=123 && indexZ<=126) || (indexZ>=148 && indexZ<=151) || indexZ==173 || indexZ==174)
		{
			// Bin que representa 1 solo sinograma
			switch(indexZ)
			{
				case 0:
					Sinograms2D[0]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i]);
					break;
				case 46:
					auxr1 = 23;
					auxr2 = 23;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i]);
					break;
				case 47:
					auxr1 = 0;
					auxr2 = 4;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i]);
					break;
				case 48:
					auxr1 = 0;
					auxr2 = 5;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i]);
					break;
				case 84:
					auxr1 = 18;
					auxr2 = 23;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i]);
					break;
				case 85:
					auxr1 = 19;
					auxr2 = 23;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i]);
					break;
				case 86:
					auxr1 = 4;
					auxr2 = 0;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i]);
					break;
				case 87:
					auxr1 = 5;
					auxr2 = 0;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i]);
					break;
				case 123:
					auxr1 = 23;
					auxr2 = 18;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i]);
					break;
				case 124:
					auxr1 = 23;
					auxr2 = 19;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i]);
					break;
				case 125:
					auxr1 = 0;
					auxr2 = 11;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i]);
					break;
				case 126:
					auxr1 = 0;
					auxr2 = 12;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i]);
					break;
				case 148:
					auxr1 = 11;
					auxr2 = 23;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i]);
					break;
				case 149:
					auxr1 = 12;
					auxr2 = 23;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i]);
					break;
				case 150:
					auxr1 = 11;
					auxr2 = 0;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i]);
					break;
				case 151:
					auxr1 = 12;
					auxr2 = 0;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i]);
					break;
				case 173:
					auxr1 = 23;
					auxr2 = 11;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i]);
					break;
				case 174:
					auxr1 = 23;
					auxr2 = 12;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i]);
					break;
			}
        }

        //all sinograms which include 2 rings
        else if (indexZ==1 || indexZ==45|| indexZ==49 || indexZ==50 || indexZ==82 || indexZ==83 ||
                indexZ==88 || indexZ==89 || indexZ==121 || indexZ==122 || indexZ==127 || indexZ==128 || indexZ==146 
                || indexZ==147 || indexZ==152 || indexZ==153 || indexZ==171 || indexZ==172)
		{
                
            // Bin que representa dos sinograma, sumo las cuentas dividido 2 para mantener las cuentas totales
			// y no desbalancear la distribucion
			switch(indexZ)
			{
				case 1:
					auxr1 = 1;
					auxr2 = 0;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 2);
					auxr1 = 0;
					auxr2 = 1;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 2);
					break;
				case 45:
					auxr1 = 23;
					auxr2 = 22;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 2);
					auxr1 = 22;
					auxr2 = 23;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 2);
					break;
				case 49:
					auxr1 = 0;
					auxr2 = 6;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 2);
					auxr1 = 1;
					auxr2 = 5;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 2);
					break;
				case 50:
					auxr1 = 1;
					auxr2 = 6;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 2);
					auxr1 = 0;
					auxr2 = 7;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 2);
					break;
				case 82:
					auxr1 = 16;
					auxr2 = 23;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 2);
					auxr1 = 17;
					auxr2 = 22;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 2);
					break;
				case 83:
					auxr1 = 17;
					auxr2 = 23;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 2);
					auxr1 = 18;
					auxr2 = 22;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 2);
					break;
				case 88:
					auxr1 = 6;
					auxr2 = 0;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 2);
					auxr1 = 5;
					auxr2 = 1;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 2);
					break;
				case 89:
					auxr1 = 7;
					auxr2 = 0;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 2);
					auxr1 = 6;
					auxr2 = 1;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 2);
					break;
				case 121:
					auxr1 = 23;
					auxr2 = 16;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 2);
					auxr1 = 22;
					auxr2 = 17;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 2);
					break;
				case 122:
					auxr1 = 23;
					auxr2 = 17;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 2);
					auxr1 = 22;
					auxr2 = 18;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 2);
					break;
				case 127:
					auxr1 = 1;
					auxr2 = 12;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 2);
					auxr1 = 0;
					auxr2 = 13;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 2);
					break;
				case 128:
					auxr1 = 1;
					auxr2 = 13;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 2);
					auxr1 = 0;
					auxr2 = 14;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 2);
					break;
				case 146:
					auxr1 = 10;
					auxr2 = 22;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 2);
					auxr1 = 9;
					auxr2 = 23;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 2);
					break;
				case 147:
					auxr1 = 11;
					auxr2 = 22;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 2);
					auxr1 = 10;
					auxr2 = 23;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 2);
					break;
				case 152:
					auxr1 = 13;
					auxr2 = 0;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 2);
					auxr1 = 12;
					auxr2 = 1;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 2);
					break;
				case 153:
					auxr1 = 14;
					auxr2 = 0;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 2);
					auxr1 = 13;
					auxr2 = 1;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 2);
					break;
				case 171:
					auxr1 = 23;
					auxr2 = 9;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 2);
					auxr1 = 22;
					auxr2 = 10;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 2);
					break;
				case 172:
					auxr1 = 23;
					auxr2 = 10;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 2);
					auxr1 = 22;
					auxr2 = 11;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 2);
					break;     
				}
        }

        //all sinograms which include 3 rings and do not follow the 3 / 4 rings pattern (see below)
        else if (indexZ==51 || indexZ==81 || indexZ==90 || indexZ==120 || indexZ==129 || indexZ==145 ||
                 indexZ==154 || indexZ==170)
		{
            switch(indexZ)
			{  
                case 51:
					auxr1 = 1;
					auxr2 = 7;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 3);
					auxr1 += 1;
					auxr2 -= 1;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 3);
					auxr1 -= 2;
					auxr2 += 2;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 3);
					break;
				case 81:
					auxr1 = 16;
					auxr2 = 22;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 3);
					auxr1 += 1;
					auxr2 -= 1;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 3);
					auxr1 -= 2;
					auxr2 += 2;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 3);
					break;
				case 90:
					auxr1 = 7;
					auxr2 = 1;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 3);
					auxr1 += 1;
					auxr2 -= 1;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 3);
					auxr1 -= 2;
					auxr2 += 2;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 3);
					break;
				case 120:
					auxr1 = 22;
					auxr2 = 16;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 3);
					auxr1 += 1;
					auxr2 -= 1;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 3);
					auxr1 -= 2;
					auxr2 += 2;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 3);
					break;
				case 129:
					auxr1 = 1;
					auxr2 = 14;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 3);
					auxr1 += 1;
					auxr2 -= 1;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 3);
					auxr1 -= 2;
					auxr2 += 2;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 3);
					break;
				case 145:
					auxr1 = 9;
					auxr2 = 22;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 3);
					auxr1 += 1;
					auxr2 -= 1;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 3);
					auxr1 -= 2;
					auxr2 += 2;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 3);
					break;
				case 154:
					auxr1 = 14;
					auxr2 = 1;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 3);
					auxr1 += 1;
					auxr2 -= 1;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 3);
					auxr1 -= 2;
					auxr2 += 2;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 3);
					break;
				case 170:
					auxr1 = 22;
					auxr2 = 9;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 3);
					auxr1 += 1;
					auxr2 -= 1;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 3);
					auxr1 -= 2;
					auxr2 += 2;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 3);
					break;
				}
        }
        
   
        //Calculations for segment 0,1,2 
        //in the middle part of the section 0,1 and 2 a 3 / 4 ring pattern could be seen,
        //so that all pair indexZ includes 3 rings and all odd indexZ includes 4 rings
        else if (indexZ<85 || (indexZ>=125 && indexZ<150))
		{
            if (indexZ%2==0)
			{
				// Sinogramas de 3
				if(indexZ<46)
				{
					//Segmento 0
					auxr1 = indexZ/2;
					auxr2 = indexZ/2;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 3);
					auxr1 += 1;
					auxr2 -= 1;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 3);
					auxr1 -= 2;
					auxr2 += 2;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 3);
				}
				else if(indexZ < 85)
				{
					// Segmento 1
					auxr1 = (indexZ - 50) /2;
					auxr2 = (indexZ - 50) /2 + 7;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 3);
					auxr1 += 1;
					auxr2 -= 1;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 3);
					auxr1 -= 2;
					auxr2 += 2;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 3);
				}
				else
				{
					//Segmento 2
					auxr1 = (indexZ - 128) /2;
					auxr2 = (indexZ - 128) /2 + 14;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 3);
					auxr1 += 1;
					auxr2 -= 1;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 3);
					auxr1 -= 2;
					auxr2 += 2;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 3);
				}
            }
            else
			{
				// Sinogramas de 4
                if(indexZ<46)
				{
					//Segmento 0
					auxr1 = (indexZ+1)/2;
					auxr2 = (indexZ+1)/2 - 1;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 4);
					auxr1 += 1;
					auxr2 -= 1;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 4);
					auxr1 -= 2;
					auxr2 += 2;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 4);
					auxr1 -= 1;
					auxr2 += 1;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 4);
				}
				else if(indexZ < 85)
				{
					// Segmento 1
					auxr1 = (indexZ - 49) /2;
					auxr2 = (indexZ - 49) /2 + 6;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 4);
					auxr1 += 1;
					auxr2 -= 1;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 4);
					auxr1 -= 2;
					auxr2 += 2;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 4);
					auxr1 -= 1;
					auxr2 += 1;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 4);
				}
				else
				{
					//Segmento 2
					auxr1 = (indexZ - 127) /2;
					auxr2 = (indexZ - 127) /2 + 13;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 4);
					auxr1 += 1;
					auxr2 -= 1;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 4);
					auxr1 -= 2;
					auxr2 += 2;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 4);
					auxr1 -= 1;
					auxr2 += 1;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 4);
				}
            }
        }
        //calculations for segment -1 and -2
        //in section -1 and -2 it is just reversed, so all pair IDs include
        // 4 rings and all odd IDs 3 rings
        else if (indexZ>=86 && indexZ<125 || indexZ >= 150 && indexZ < 175)
		{ 
            if (indexZ%2!=0)
			{
				// Sinoramas de 3
				// Sinogramas de 3
				if(indexZ<125)
				{
					//Segmento -1
					auxr1 = (indexZ - 89) /2 + 7;
					auxr2 = (indexZ - 89) /2;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 3);
					auxr1 += 1;
					auxr2 -= 1;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 3);
					auxr1 -= 2;
					auxr2 += 2;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 3);
				}
				else if(indexZ < 175)
				{
					// Segmento -2
					auxr1 = (indexZ - 153) /2 + 14;
					auxr2 = (indexZ - 153) /2;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 3);
					auxr1 += 1;
					auxr2 -= 1;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 3);
					auxr1 -= 2;
					auxr2 += 2;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 3);
				}
            }
            else
			{
				//Sinogramas de 4
                if(indexZ<125)
				{
					//Segmento -1
					auxr1 = (indexZ - 90) /2 + 8;
					auxr2 = (indexZ - 90) /2;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 4);
					auxr1 += 1;
					auxr2 -= 1;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 4);
					auxr1 -= 2;
					auxr2 += 2;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 4);
					auxr1 -= 1;
					auxr2 += 1;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 4);
				}
				else if(indexZ < 175)
				{
					// Segmento -2
					auxr1 = (indexZ - 154) /2 + 15;
					auxr2 = (indexZ - 154) /2;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 4);
					auxr1 += 1;
					auxr2 -= 1;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 4);
					auxr1 -= 2;
					auxr2 += 2;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 4);
					auxr1 -= 1;
					auxr2 += 1;
					Sinograms2D[auxr2 * NZ + auxr1]->setSinogramBin(indexPhi,indexR,MichelogramBio16[i] / 4);
				}
            } 
        }
		//Depending of the events hay have differents og LORs
	}

	fclose(fid);
	free(DatosLista);
	return true;
}


bool Michelogram::ReadDataFromSinogram3D(Sinogram3D* MiSino3D)
{
	/// Se pasan los datos del Sinogram3D al Michelograma, para esto se recorren todos los
	/// segmentos y se suman las cuentas correspondientes al bin al par de anillos correspondientes
	/// en el Michelograma.
	/* Estas variables deber�an ser iguales, la deber�amos asignar si este fuera un cosntructor, para repensar
	NZ = MiSino3D->NRings;
	NR = MiSino3D->NR;
	*/
	RFOV = MiSino3D->getRadioFov_mm();
	ZFOV = MiSino3D->getAxialFoV_mm();
	/// Chequeo que el Sino3D tenga el mismo tama�o que el Micho, sino salgo con false.
	if((NR != MiSino3D->getNumR())||(NProj != MiSino3D->getNumProj())||(NZ != MiSino3D->getNumRings()))
	{
		sprintf(Error, "El tama�o del Sinograma3D no coincide con el del Michelograma que se quiere cargar");
		return false;
	}
	for(int i = 0;  i < MiSino3D->getNumSegments(); i++)
	{
		for(int j = 0; j < MiSino3D->getSegment(i)->getNumSinograms(); j++)
		{
			for(int k = 0; k < MiSino3D->getSegment(i)->getSinogram2D(j)->getNumZ(); k++)
			{
				/// El indice del Michelograma dentro del array de Sinos2D se recorre
				/// a través de la variable Ring1. O sea indice = iRing1 + iRing2 * NRing
				int indiceMicho = MiSino3D->getSegment(i)->getSinogram2D(j)->getRing1FromList(k) + NZ * MiSino3D->getSegment(i)->getSinogram2D(j)->getRing2FromList(k);
				Sinograms2D[indiceMicho]->setRing1(MiSino3D->getSegment(i)->getSinogram2D(j)->getRing1FromList(k));
				Sinograms2D[indiceMicho]->setRing2(MiSino3D->getSegment(i)->getSinogram2D(j)->getRing2FromList(k));
				for(int l = 0; l < NProj; l++)
				{
				  for(int m = 0; m < NR; m++)
				  {
					Sinograms2D[indiceMicho]->incrementSinogramBin(l,m, MiSino3D->getSegment(i)->getSinogram2D(j)->getSinogramBin(l,m));
				  }
				}
			}
		}
	}
	return true;
}


void FillValues(MichelogramValues* MyMichelogramValues, SizeMichelogram MySizeMichelogram)
{
// Initialization of Z values
	// Allocates Memory for the value's vectors
	MyMichelogramValues->PhiValues = (float*) malloc(MySizeMichelogram.NProj*sizeof(float));
	MyMichelogramValues->RValues = (float*) malloc(MySizeMichelogram.NR*sizeof(float));
	MyMichelogramValues->ZValues = (float*) malloc(MySizeMichelogram.NZ*sizeof(float));
	float ZIncrement = (float)MySizeMichelogram.ZFOV/MySizeMichelogram.NZ;
	for(int i = 0; i < MySizeMichelogram.NZ; i ++)
	{
		// Initialization of Z Values
		MyMichelogramValues->ZValues[i] = ZIncrement/2 + i * ZIncrement;
	}
	// Initialization
	float RIncrement = (2 * MySizeMichelogram.RFOV) / MySizeMichelogram.NR;
	float PhiIncrement = (float)MySizeMichelogram.MAXANG / MySizeMichelogram.NProj;
	for(int i = 0; i < MySizeMichelogram.NProj; i ++)
	{
		// Initialization of Phi Values
		MyMichelogramValues->PhiValues[i] = PhiIncrement/2 + i * PhiIncrement;
	}
	for(int j = 0; j < MySizeMichelogram.NR; j++)
	{
		MyMichelogramValues->RValues[j] = RIncrement/2 + j * RIncrement - MySizeMichelogram.RFOV;
	}
}

bool Michelogram::FillConstant(float Value)
{
	/// Se llena todos los bins del sinograma con un valor constante de valor Value.
	/// Esto puede ser de utilidad para calcular el sensibility volume.
	for(int i=0; i< NZ*NZ; i++)
	{
		for(int k=0; k < NProj; k++)
		{
			for(int l=0; l < NR; l++)
			{
				Sinograms2D[i]->setSinogramBin(k, l, Value);
			}
		}
	}
	return true;
}
