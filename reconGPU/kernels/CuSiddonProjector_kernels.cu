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
#include "../kernels/CuSiddonWithSurfaces.cu"

// Al version with only 1 sample
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
  if (iProj>numProj/2)
	  iR = iR-1;
  //CUDA_GetPointsFromLOR(d_thetaValues_deg[iProj], d_RValues_mm[iR], d_ring1[blockIdx.y], d_ring2[blockIdx.y], d_RadioScanner_mm, &P1, &P2);
  CUDA_GetPointsFromBinsMmr (d_thetaValues_deg[iProj], iR, numR, d_ring1[blockIdx.y], d_ring2[blockIdx.y], d_RadioScanner_mm, &P1, &P2);
  LOR.x = P2.x - P1.x;
  LOR.y = P2.y - P1.y;
  LOR.z = P2.z - P1.z;
  //if((blockIdx.y == 0)&&(iR==110)&&(iProj==0))
  //  printf("Points: %f,%f,%f\t%f,%f,%f\t bin value:%f\n", P1.x, P1.y, P1.z, P2.x, P2.y, P2.z, michelogram[iBin]);
  
  cuSiddonWithTextures (&LOR, &P1, volume, michelogram, iBin);
}

__global__ void cuSiddonDivideAndBackproject(float* d_inputSinogram, float* d_estimatedSinogram, float* d_outputImage, 
					     float *d_ring1, float *d_ring2, int numR, int numProj, int numRings, int numSinos)
{
  float4 P1;
  float4 P2;
  float4 LOR;
  /// Calculo dentro del sinograma 2D, se obtiene con threadIdx.x y blockIdx.x.
  int iBin =  threadIdx.x + (blockIdx.x * blockDim.x);
  // First compare inside the sinogram:
  if(iBin>= d_numBinsSino2d)
    return;
  int iR = iBin % numR;
  int iProj = (int)((float)iBin / (float)numR);
  iBin = iBin + blockIdx.y * d_numBinsSino2d;
if (iProj>numProj/2)
	  iR = iR-1;
  // Primero hago la división:
  if(d_estimatedSinogram[iBin]!=0)
  {
    d_estimatedSinogram[iBin] = d_inputSinogram[iBin] / d_estimatedSinogram[iBin];
    // Después backprojection:
    //CUDA_GetPointsFromLOR(d_thetaValues_deg[iProj], d_RValues_mm[iR], d_ring1[blockIdx.y], d_ring2[blockIdx.y], d_RadioScanner_mm, &P1, &P2);
	CUDA_GetPointsFromBinsMmr (d_thetaValues_deg[iProj], iR, numR, d_ring1[blockIdx.y], d_ring2[blockIdx.y], d_RadioScanner_mm, &P1, &P2);
    LOR.x = P2.x - P1.x;
    LOR.y = P2.y - P1.y;
    LOR.z = P2.z - P1.z;
    CuSiddonBackprojection(&LOR, &P1, d_outputImage, d_estimatedSinogram, iBin);
  }
  // If denominator 0, or inputsinogram 0, dont do anything (backroject of a zero)
  /*else if(d_inputSinogram[iBin] != 0)
  {
    /// Los bins de los sinogramas Input y Estimates son 0, o sea tengo el valor indeterminado 0/0.
    /// Lo más lógico pareciera ser dejarlo en 0 al cociente, para que no sume al backprojection.
    /// Sumarle 0 es lo mismo que nada.
    d_estimatedSinogram[indiceMichelogram] = 0;
  }
  */
  
}

__global__ void cuSiddonBackprojection(float* d_inputSinogram, float* d_outputImage, 
				       float *d_ring1_mm, float *d_ring2_mm, int numR, int numProj, int numRings, int numSinos)
{
  float4 P1;
  float4 P2;
  float4 LOR;
  /// Calculo dentro del sinograma 2D, se obtiene con threadIdx.x y blockIdx.x.
  int iBin =  threadIdx.x + (blockIdx.x * blockDim.x);
  // First compare inside the sinogram:
  if(iBin>= d_numBinsSino2d)
    return;
  int iR = iBin % numR;
  int iProj = (int)((float)iBin / (float)numR);
  iBin = iBin + blockIdx.y * d_numBinsSino2d;
  if (iProj>numProj/2)
	  iR = iR-1;
  if(d_inputSinogram[iBin] != 0)
  {
    //CUDA_GetPointsFromLOR(d_thetaValues_deg[iProj], d_RValues_mm[iR], d_ring1_mm[blockIdx.y], d_ring2_mm[blockIdx.y], d_RadioScanner_mm, &P1, &P2);
	CUDA_GetPointsFromBinsMmr (d_thetaValues_deg[iProj], iR, numR, d_ring1_mm[blockIdx.y], d_ring2_mm[blockIdx.y], d_RadioScanner_mm, &P1, &P2);
    LOR.x = P2.x - P1.x;
    LOR.y = P2.y - P1.y;
    LOR.z = P2.z - P1.z;
    CuSiddonBackprojection(&LOR, &P1, d_outputImage, d_inputSinogram, iBin);
  }
}

// Al version with oversample, it could have been done all in one. But I didn't want to add any overhead or bug to the standard version.
__global__ void cuSiddonOversampledProjection (float* volume, float* michelogram, float *d_ring1, float *d_ring2, float ringWidth_mm, int numR, int numSamples, int numAxialSamples)
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
  // axial step:
  float deltaZ = ringWidth_mm/numAxialSamples;
  // Repeat the Siddon for all the samples, I need to establish the limits for each LOR that are not evenly separated, for that I use the midpoint with the neighourgh samples on each side:
  float rLimInf, rLimSup;
  // Lower limit for this LOR
  if (iR == 0)
	  rLimInf = d_RValues_mm[iR] - (d_RValues_mm[iR+1]-d_RValues_mm[iR])/2;
  else
	  rLimInf = (d_RValues_mm[iR] + d_RValues_mm[iR-1])/2;
  // Uper limit for this LOR
  if (iR == numR-1)
	  rLimSup = d_RValues_mm[iR] + (d_RValues_mm[iR]-d_RValues_mm[iR-1])/2; // Use the same distance to the previous sample
  else
	  rLimSup = (d_RValues_mm[iR] + d_RValues_mm[iR+1])/2;
  // In order to use less registers, I reuse the variables:
  // rLimSup: is now deltR, the distance between samples
  // rLimInf: is now r, the current  sample value.
  rLimSup = (rLimSup - rLimInf)/numSamples; //float deltaR = (rLimSup - rLimInf)/numSamples;
  rLimInf = rLimInf + rLimSup/2; // float r = rLimInf + deltaR/2;
  for(int i = 0; i < numSamples; i++)
  {
	  for(int j = 0; j < numAxialSamples; j++)
	  {
		// The r coordinate is in rLimInf  
		CUDA_GetPointsFromLOR(d_thetaValues_deg[iProj], rLimInf, d_ring1[blockIdx.y]-ringWidth_mm/2+deltaZ/2+j*deltaZ, d_ring2[blockIdx.y]-ringWidth_mm/2+deltaZ/2+j*deltaZ, d_RadioScanner_mm, &P1, &P2);
		//CUDA_GetPointsFromLOR(d_thetaValues_deg[iProj], rLimInf, d_ring1[blockIdx.y], d_ring2[blockIdx.y], d_RadioScanner_mm, &P1, &P2);
		LOR.x = P2.x - P1.x;
		LOR.y = P2.y - P1.y;
		LOR.z = P2.z - P1.z;
		cuSiddonWithTextures (&LOR, &P1, volume, michelogram, iBin);
	  }
	// Update the r for this sample:
	rLimInf += rLimSup; // r = r + deltaR;
  }
}

__global__ void cuSiddonOversampledDivideAndBackproject(float* d_inputSinogram, float* d_estimatedSinogram, float* d_outputImage, 
					     float *d_ring1, float *d_ring2, float ringWidth_mm, int numR, int numSamples, int numAxialSamples)
{
  float4 P1;
  float4 P2;
  float4 LOR;
  /// Calculo dentro del sinograma 2D, se obtiene con threadIdx.x y blockIdx.x.
  int iBin =  threadIdx.x + (blockIdx.x * blockDim.x);
  // First compare inside the sinogram:
  if(iBin>= d_numBinsSino2d)
    return;
  int iR = iBin % numR;
  int iProj = (int)((float)iBin / (float)numR);
  // axial step:
  float deltaZ = ringWidth_mm/numAxialSamples;
  iBin = iBin + blockIdx.y * d_numBinsSino2d;

  // Primero hago la división:
  if(d_estimatedSinogram[iBin]!=0)
  {
    d_estimatedSinogram[iBin] = d_inputSinogram[iBin] / d_estimatedSinogram[iBin];
	
	// Repeat the Siddon for all the samples, I need to establish the limits for each LOR that are not evenly separated, for that I use the midpoint with the neighourgh samples on each side:
	float rLimInf, rLimSup;
	// Lower limit for this LOR
	if (iR == 0)
		rLimInf = 0;
	else
		rLimInf = (d_RValues_mm[iR] + d_RValues_mm[iR-1])/2;
	// Uper limit for this LOR
	if (iR == numR-1)
		rLimSup = d_RValues_mm[iR] + (d_RValues_mm[iR]-d_RValues_mm[iR-1])/2; // Use the same distance to the previous sample
	else
		rLimSup = (d_RValues_mm[iR] + d_RValues_mm[iR+1])/2;
	// In order to use less registers, I reuse the variables:
	// rLimSup: is now deltR, the distance between samples
	// rLimInf: is now r, the current  sample value.
	rLimSup = (rLimSup - rLimInf)/numSamples; //float deltaR = (rLimSup - rLimInf)/numSamples;
	rLimInf = rLimInf + rLimSup/2; // float r = rLimInf + deltaR/2;
  
    // Después backprojection:
	for(int i = 0; i < numSamples; i++)
	{
		for(int j = 0; j < numAxialSamples; j++)
		{
			// The r coordinate is in rLimInf  
			CUDA_GetPointsFromLOR(d_thetaValues_deg[iProj], rLimInf, d_ring1[blockIdx.y]-ringWidth_mm/2+deltaZ/2+j*deltaZ, d_ring2[blockIdx.y]-ringWidth_mm/2+deltaZ/2+j*deltaZ, d_RadioScanner_mm, &P1, &P2);
			//CUDA_GetPointsFromLOR(d_thetaValues_deg[iProj], rLimInf, d_ring1[blockIdx.y], d_ring2[blockIdx.y], d_RadioScanner_mm, &P1, &P2);
			LOR.x = P2.x - P1.x;
			LOR.y = P2.y - P1.y;
			LOR.z = P2.z - P1.z;
			CuSiddonBackprojection(&LOR, &P1, d_outputImage, d_estimatedSinogram, iBin);
		}
		// Update the r for this sample:
		rLimInf += rLimSup; // r = r + deltaR;		
	}
  }
  // If denominator 0, or inputsinogram 0, dont do anything (backroject of a zero)
  /*else if(d_inputSinogram[iBin] != 0)
  {
    /// Los bins de los sinogramas Input y Estimates son 0, o sea tengo el valor indeterminado 0/0.
    /// Lo más lógico pareciera ser dejarlo en 0 al cociente, para que no sume al backprojection.
    /// Sumarle 0 es lo mismo que nada.
    d_estimatedSinogram[indiceMichelogram] = 0;
  }
  */
}

__global__ void cuSiddonOversampledBackprojection(float* d_inputSinogram, float* d_outputImage, 
				       float *d_ring1_mm, float *d_ring2_mm, float ringWidth_mm, int numR, int numSamples, int numAxialSamples)
{
  float4 P1;
  float4 P2;
  float4 LOR;
  /// Calculo dentro del sinograma 2D, se obtiene con threadIdx.x y blockIdx.x.
  int iBin =  threadIdx.x + (blockIdx.x * blockDim.x);
  // First compare inside the sinogram:
  if(iBin>= d_numBinsSino2d)
    return;
  int iR = iBin % numR;
  int iProj = (int)((float)iBin / (float)numR);
  // axial step:
  float deltaZ = ringWidth_mm/numAxialSamples;
  iBin = iBin + blockIdx.y * d_numBinsSino2d;
  
  if(d_inputSinogram[iBin] != 0)
  {
	// Repeat the Siddon for all the samples, I need to establish the limits for each LOR that are not evenly separated, for that I use the midpoint with the neighourgh samples on each side:
	float rLimInf, rLimSup;
	// Lower limit for this LOR
	if (iR == 0)
		rLimInf = d_RValues_mm[iR] - (d_RValues_mm[iR+1]-d_RValues_mm[iR])/2;
	else
		rLimInf = (d_RValues_mm[iR] + d_RValues_mm[iR-1])/2;
	// Uper limit for this LOR
	if (iR == numR-1)
		rLimSup = d_RValues_mm[iR] + (d_RValues_mm[iR]-d_RValues_mm[iR-1])/2; // Use the same distance to the previous sample
	else
		rLimSup = (d_RValues_mm[iR] + d_RValues_mm[iR+1])/2;
	// In order to use less registers, I reuse the variables:
	// rLimSup: is now deltR, the distance between samples
	// rLimInf: is now r, the current  sample value.
	rLimSup = (rLimSup - rLimInf)/numSamples; //float deltaR = (rLimSup - rLimInf)/numSamples;
	rLimInf = rLimInf + rLimSup/2; // float r = rLimInf + deltaR/2;
	for(int i = 0; i < numSamples; i++)
	{
		for(int j = 0; j < numAxialSamples; j++)
		{
			// The r coordinate is in rLimInf  
			CUDA_GetPointsFromLOR(d_thetaValues_deg[iProj], rLimInf, d_ring1_mm[blockIdx.y]-ringWidth_mm/2+deltaZ/2+j*deltaZ, d_ring2_mm[blockIdx.y]-ringWidth_mm/2+deltaZ/2+j*deltaZ, d_RadioScanner_mm, &P1, &P2);
			//CUDA_GetPointsFromLOR(d_thetaValues_deg[iProj], rLimInf, d_ring1_mm[blockIdx.y], d_ring2_mm[blockIdx.y], d_RadioScanner_mm, &P1, &P2);
			LOR.x = P2.x - P1.x;
			LOR.y = P2.y - P1.y;
			LOR.z = P2.z - P1.z;
			CuSiddonBackprojection(&LOR, &P1, d_outputImage, d_inputSinogram, iBin);
		}
		// Update the r for this sample:
		rLimInf += rLimSup; // r = r + deltaR;
	}
  }
}




/// El ángulo de GetPointsFromLOR debe estar en radianes.
__device__ void CUDA_GetPointsFromLOR (float PhiAngle, float r, float Z1, float Z2, float cudaRscanner, float4* P1, float4* P2)
{
  float sinValue, cosValue;
  sincosf(PhiAngle*DEG_TO_RAD, &sinValue, &cosValue);
  float auxValue = sqrtf((cudaRscanner) * (cudaRscanner) - r * r);
  P1->x = r * cosValue + sinValue * auxValue;
  P1->y = r * sinValue - cosValue * auxValue;
  P1->z = (Z1+Z2)/2.0f - (Z2-Z1)/(2.0f*cudaRscanner+d_crystalElementLength_mm)*cudaRscanner - (Z2-Z1)/(2.0f*cudaRscanner+d_crystalElementLength_mm)*d_meanDOI_mm;
  P2->x = r * cosValue - sinValue * auxValue;
  P2->y = r * sinValue + cosValue * auxValue;
  P2->z = (Z1+Z2)/2.0f + (Z2-Z1)/(2.0f*cudaRscanner+d_crystalElementLength_mm)*cudaRscanner + (Z2-Z1)/(2.0f*cudaRscanner+d_crystalElementLength_mm)*d_meanDOI_mm;
}

/// El ángulo de GetPointsFromLOR debe estar en radianes.
__device__ void CUDA_GetPointsFromBinsMmr (float PhiAngle, int iR, int numR, float Z1, float Z2, float cudaRscanner, float4* P1, float4* P2)
{
  float sinValue, cosValue, r, lr;
  if (PhiAngle < 90)
	lr = -d_binSize_mm/2 + (d_binSize_mm*(iR+1-(float)(numR/2)));
  else
	lr = (d_binSize_mm*(iR+1-(float)(numR/2)));  
  r = (cudaRscanner + d_meanDOI_mm* cos(lr/cudaRscanner)) * sin(lr/cudaRscanner);
  sincosf(PhiAngle*DEG_TO_RAD, &sinValue, &cosValue);
  float auxValue = sqrtf((cudaRscanner) * (cudaRscanner) - r * r);
  P1->x = r * cosValue + sinValue * auxValue;
  P1->y = r * sinValue - cosValue * auxValue;
  P1->z = (Z1+Z2)/2.0f - (Z2-Z1)/(2.0f*cudaRscanner+d_crystalElementLength_mm)*cudaRscanner - (Z2-Z1)/(2.0f*cudaRscanner+d_crystalElementLength_mm)*d_meanDOI_mm;
  P2->x = r * cosValue - sinValue * auxValue;
  P2->y = r * sinValue + cosValue * auxValue;
  P2->z = (Z1+Z2)/2.0f + (Z2-Z1)/(2.0f*cudaRscanner+d_crystalElementLength_mm)*cudaRscanner + (Z2-Z1)/(2.0f*cudaRscanner+d_crystalElementLength_mm)*d_meanDOI_mm;
}

#endif