#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <Siddon.h>
#include <TriSiddon.h>
#include <Geometry.h>
#include <Images.h>
#include <iostream>
#include <limits>

using namespace::std;	

// This function calculates TriSiddon Wieghts for a lor. It gets as parameters, the LOR in
// a Line3D object which P0 is the P1 of the LOR, the values of the planes in X, Y, Z, and a double pointer
// where all the wieghts will be loaded. It's a double pointer, because is a dynamic array, so the adress
// of the array can change when reallocating memory. In order to not loose the reference in the calling
// function we use a double pointer.
void TriSiddon (float Phi, float R, float Z1, float Z2, Image* image, SiddonSegment** WeightsList, unsigned int* LengthList)
{
	// El algoritmo TriSiddon calcula para cada TOR(tube of response) un valor representativo de la sensibilidad 
	// geométrica de una TOR respecto de los píxeles, dividiendo dicho TOR en 3 LOR equiespaciadas, y sumando los 
	// valores de Siddon en cada pixel para cada una de las 3 LORS y ponderándolos según el peso de cada LOR en el TOR. 
	Point3D P1, P2;
	Line3D LOR;
	SiddonSegment** AuxList1;	// Lista de segmentos Siddon para LOR central
 	SiddonSegment** AuxList2;	// Lista de segmentos Siddon para LOR lateral + deltaR/3
	SiddonSegment** AuxList3;	// Lista de segmentos Siddon para LOR lateral - deltaR/3
	SiddonSegment** AuxList4;	// Lista de segmentos Siddon para LOR lateral + deltaR/3
	SiddonSegment** AuxList5;	// Lista de segmentos Siddon para LOR lateral - deltaR/3
	int LengthList1, LengthList2, LengthList3, LengthList4, LengthList5;
	
	//Inicializo los 3 punteros a array
	AuxList1 = (SiddonSegment**) malloc(sizeof(SiddonSegment*));
	AuxList2 = (SiddonSegment**) malloc(sizeof(SiddonSegment*));
	AuxList3 = (SiddonSegment**) malloc(sizeof(SiddonSegment*));
	AuxList4 = (SiddonSegment**) malloc(sizeof(SiddonSegment*));
	AuxList5 = (SiddonSegment**) malloc(sizeof(SiddonSegment*));

	// Primero la lor central 
	GetPointsFromLOR(Phi, R, Z1, Z2, RSCANNER, &P1, &P2);
	LOR.P0 = P1;
	LOR.Vx = P2.X - P1.X;
	LOR.Vy = P2.Y - P1.Y;
	LOR.Vz = P2.Z - P1.Z;
	Siddon(LOR, image, AuxList1, &LengthList1, PESO_CENTRAL);
	
	// Ahora las dos lors a los costados, que se encuentran desplazadas +-deltaR/3
	GetPointsFromLOR(Phi, R+DELTA_R/3, Z1, Z2, RSCANNER, &P1, &P2);
	LOR.P0 = P1;
	LOR.Vx = P2.X - P1.X;
	LOR.Vy = P2.Y - P1.Y;
	LOR.Vz = P2.Z - P1.Z;
	Siddon(LOR, image, AuxList2, &LengthList2, PESO_LATERAL);

	GetPointsFromLOR(Phi, R-DELTA_R/3, Z1, Z2, RSCANNER, &P1, &P2);
	LOR.P0 = P1;
	LOR.Vx = P2.X - P1.X;
	LOR.Vy = P2.Y - P1.Y;
	LOR.Vz = P2.Z - P1.Z;
	Siddon(LOR, image, AuxList3, &LengthList3, PESO_LATERAL);

	GetPointsFromLOR(Phi, R, Z1-DELTA_Z/3, Z2-DELTA_Z/3, RSCANNER, &P1, &P2);
	LOR.P0 = P1;
	LOR.Vx = P2.X - P1.X;
	LOR.Vy = P2.Y - P1.Y;
	LOR.Vz = P2.Z - P1.Z;
	Siddon(LOR, image, AuxList4, &LengthList4, PESO_LATERAL);

	GetPointsFromLOR(Phi, R, Z1+DELTA_Z/3, Z2+DELTA_Z/3, RSCANNER, &P1, &P2);
	LOR.P0 = P1;
	LOR.Vx = P2.X - P1.X;
	LOR.Vy = P2.Y - P1.Y;
	LOR.Vz = P2.Z - P1.Z;
	Siddon(LOR, image, AuxList5, &LengthList5, PESO_LATERAL);
	
	// Hago una lista que junte las 3 listas 
	LengthList[0] = LengthList1 + LengthList2 + LengthList3 + LengthList4 + LengthList5;
	WeightsList[0] = (SiddonSegment*) malloc(sizeof(SiddonSegment) * LengthList[0]);
	memcpy((void*)(WeightsList[0]), AuxList1[0], sizeof(SiddonSegment)*LengthList1);
	memcpy((void*)(WeightsList[0]+LengthList1), AuxList2[0], sizeof(SiddonSegment)*LengthList2);
	memcpy((void*)(WeightsList[0]+LengthList1+LengthList2), AuxList3[0], sizeof(SiddonSegment)*LengthList3);
	memcpy((void*)(WeightsList[0]+LengthList1+LengthList2+LengthList3), AuxList4[0], sizeof(SiddonSegment)*LengthList4);
	memcpy((void*)(WeightsList[0]+LengthList1+LengthList2+LengthList3+LengthList4), AuxList5[0], sizeof(SiddonSegment)*LengthList5);
	// Libero la memoria
	free(AuxList1[0]);
	free(AuxList2[0]);
	free(AuxList3[0]);
	free(AuxList4[0]);
	free(AuxList5[0]);
	free(AuxList1);
	free(AuxList2);
	free(AuxList3);
	free(AuxList4);
	free(AuxList5);
}
