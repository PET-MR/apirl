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
#include <vector_types.h>
#include <stdio.h>

#include <CuSiddon.h>
#include <CuSiddonProjector.h>
#include "../kernels/CuSiddon.cu"
#include "../kernels/CuSiddonWithTextures.cu"

__global__ void cuSiddonProjection (float* volume, float* michelogram, float *d_ring1, float *d_ring2, int numR, int numProj, int numRings, int numSinos)
{
  int iBin =  threadIdx.x + (blockIdx.x * blockDim.x);
  // First compare inside the sinogram:
  if(iBin>= d_numBinsSino2d)
    return;
  int iR = iBin % numR;
  int iProj = (int)((float)iBin / (float)numR);
  //int iSino = blockIdx.y;	// Indice de sinograma, no genero un registro adicional al pedo
  float4 P1;// = make_float4(0,0,0);
  float4 P2;
  float4 LOR;
  iBin = iBin + blockIdx.y * d_numBinsSino2d;
  CUDA_GetPointsFromLOR(d_thetaValues_deg[iProj], d_RValues_mm[iR], d_ring1[blockIdx.y], d_ring2[blockIdx.y], d_RadioScanner_mm, &P1, &P2);
  LOR.x = P2.x - P1.x;
  LOR.y = P2.y - P1.y;
  LOR.z = P2.z - P1.z;
  cuSiddonWithTextures (&LOR, &P1, volume, michelogram, iBin);
}

__global__ void cuSiddonDivideAndBackproject(float* d_inputSinogram, float* d_estimatedSinogram, float* d_outputImage, 
					     float *d_ring1, float *d_ring2, int numR, int numProj, int numRings, int numSinos)
{
  float4 P1;
  float4 P2;
  float4 LOR;
  /// Calculo dentro del sinograma 2D, se obtiene con threadIdx.x y blockIdx.x.
  int iBin2d =  threadIdx.x + (blockIdx.x * blockDim.x);
  if(iBin2d >= (numR*numProj))
    return;
  int iR = iBin2d % numR;

  int iProj = (int)((float)iBin2d / (float)numR);
  int indiceMichelogram = iBin2d + blockIdx.y * (numProj * numR);

  // Primero hago la división:
  if(d_estimatedSinogram[indiceMichelogram]!=0)
    d_estimatedSinogram[indiceMichelogram] = d_inputSinogram[indiceMichelogram] / d_estimatedSinogram[indiceMichelogram];
  else if(d_inputSinogram[indiceMichelogram] != 0)
  {
    /// Los bins de los sinogramas Input y Estimates son 0, o sea tengo el valor indeterminado 0/0.
    /// Lo más lógico pareciera ser dejarlo en 0 al cociente, para que no sume al backprojection.
    /// Sumarle 0 es lo mismo que nada.
    d_estimatedSinogram[indiceMichelogram] = 0;
  }

  // Después backprojection:
  CUDA_GetPointsFromLOR(d_thetaValues_deg[iProj], d_RValues_mm[iR], d_ring1[blockIdx.y], d_ring2[blockIdx.y], d_RadioScanner_mm, &P1, &P2);
  LOR.x = P2.x - P1.x;
  LOR.y = P2.y - P1.y;
  LOR.z = P2.z - P1.z;
  CUDA_Siddon (&LOR, &P1, d_estimatedSinogram, d_outputImage, BACKPROJECTION, indiceMichelogram);
}

__global__ void cuSiddonBackprojection(float* d_inputSinogram, float* d_outputImage, 
				       float *d_ring1_mm, float *d_ring2_mm, int numR, int numProj, int numRings, int numSinos)
{
  float4 P1;
  float4 P2;
  float4 LOR;
  /// Calculo dentro del sinograma 2D, se obtiene con threadIdx.x y blockIdx.x.
  int iBin2d =  threadIdx.x + (blockIdx.x * blockDim.x);
  if(iBin2d >= (numR*numProj))
    return;
  int iR = iBin2d % numR;
  int iProj = (int)((float)iBin2d / (float)numR);
  int indiceMichelogram = iBin2d + blockIdx.y * (numProj * numR);
  
  CUDA_GetPointsFromLOR(d_thetaValues_deg[iProj], d_RValues_mm[iR], d_ring1_mm[blockIdx.y], d_ring2_mm[blockIdx.y], d_RadioScanner_mm, &P1, &P2);
  if((blockIdx.y == 0))
    printf("BlockIdx.y:%d indMichelogram:%d P1:%f,%f,%f P2:%f,%f,%f.\n", blockIdx.y, indiceMichelogram, P1.x, P1.y, P1.z, P2.x,P2.y,P2.z);
  //CUDA_GetPointsFromLOR(d_thetaValues_deg[iProj], d_RValues_mm[iR], d_ring1[blockIdx.y], d_ring2[blockIdx.y], d_RadioScanner_mm, &P1, &P2);
  LOR.x = P2.x - P1.x;
  LOR.y = P2.y - P1.y;
  LOR.z = P2.z - P1.z;
  CUDA_Siddon (&LOR, &P1, d_inputSinogram, d_outputImage, BACKPROJECTION, indiceMichelogram);
}

/// El ángulo de GetPointsFromLOR debe estar en radianes.
__device__ void CUDA_GetPointsFromLOR (float PhiAngle, float r, float Z1, float Z2, float cudaRscanner, float4* P1, float4* P2)
{
  float auxValue = sqrtf(cudaRscanner * cudaRscanner - r * r);
  float sinValue, cosValue;
  sincosf(PhiAngle*DEG_TO_RAD, &sinValue, &cosValue);
  P1->x = r * cosValue + sinValue * auxValue;
  P1->y = r * sinValue - cosValue * auxValue;
  P1->z = Z1;
  P2->x = r * cosValue - sinValue * auxValue;
  P2->y = r * sinValue + cosValue * auxValue;
  P2->z = Z2;
}

#endif