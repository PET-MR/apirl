/**
	\file CuSiddonProjector_kernels.cu
	\brief Archivo que contiene los kernels utilizados en la clase CuSiddonProjector.
	
	\todo 
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2014.07.11
	\version 1.1.0
*/

#ifndef _CUSIDDONKERNELS_H_
#define _CUSIDDONKERNELS_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <CUDA_Siddon.h>
#include "../src/CUDA_Siddon.cu"
#define MAX_PHI_VALUES	512	// Máxima cantidad de valores en el angulo theta que puede admitir la implementación.
#define MAX_R_VALUES	512	// Idem para R.
#define MAX_Z_VALUES	92	// Idem para anillos (z)
#define MAX_SPAN	7	// Máximo valor de combinación de anillos por sinograma 2D.

// Memoria constante con los valores de los angulos de la proyeccion,
__device__ __constant__ float d_thetaValues_deg[MAX_PHI_VALUES];

// Memoria constante con los valores de la distancia r.
__device__ __constant__ float d_RValues_mm[MAX_R_VALUES];

// Memoria constante con los valores de la coordenada axial o z.
__device__ __constant__ float d_AxialValues_mm[MAX_Z_VALUES];

__device__ __constant__ float d_RadioFov_mm;

__device__ __constant__ float d_AxialFov_mm;

__device__ __constant__ float d_RadioScanner_mm;

__device__ __constant__ SizeImage d_imageSize;
	
__global__ void CUDA_Forward_Projection (float* volume, float* michelogram, float* michelogram_measured, int numR, int numProj, int numRings, int numSinos)
{
  int indexSino2D =  threadIdx.x + (blockIdx.x * blockDim.x);
  if(indexSino2D>= (numR*numProj))
    return;
  int iR = indexSino2D % numR;
  int iProj = (int)((float)indexSino2D / (float)numR);
  int iZ = blockIdx.y;	// Block index y, index of the 2d sinogram, from which it can be taken  position Z1 and Z2 of th sinogram2d
  int indexRing1 = iZ%numRings; //Ring 1 : Las columnas;
  int indexRing2 = (int)(iZ/numRings);	// Ring 2 las filas	
  float4 P1;// = make_float4(0,0,0);
  float4 P2;
  float4 LOR;
  int indiceMichelogram = iR + iProj * numR + iZ * (numProj * numR);
  if(michelogram_measured[indiceMichelogram] != 0)
  {
    CUDA_GetPointsFromLOR(d_thetaValues_deg[iProj], d_RValues_mm[iR], d_AxialValues_mm[indexRing1], d_AxialValues_mm[indexRing2], d_RadioScanner_mm, &P1, &P2);
    LOR.x = P2.x - P1.x;
    LOR.y = P2.y - P1.y;
    LOR.z = P2.z - P1.z;
    CUDA_Siddon (&LOR, &P1, volume, michelogram, PROJECTION, indiceMichelogram);
  }	
}

/// El ángulo de GetPointsFromLOR debe estar en radianes.
__device__ void CUDA_GetPointsFromLOR (float PhiAngle, float r, float Z1, float Z2, float cudaRscanner, float4* P1, float4* P2)
{
	float auxValue = sqrtf(cudaRscanner * cudaRscanner - r * r);
	float sinValue, cosValue;
	sincosf(PhiAngle, &sinValue, &cosValue);
	P1->x = r * cosValue + sinValue * auxValue;
	P1->y = r * sinValue - cosValue * auxValue;
	P1->z = Z1;
	P2->x = r * cosValue - sinValue * auxValue;
	P2->y = r * sinValue + cosValue * auxValue;
	P2->z = Z2;
}

#endif