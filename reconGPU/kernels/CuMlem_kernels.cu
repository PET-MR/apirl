/**
	\file CuMlem_kernels.cu
	\brief Archivo que contiene los kernels utilizados en las clases relacionadas con Mlem.
	Los kernels son para la actualización de píxels y el cálculo de likelihood.
	
	\todo En el futuro deberían estar los kernels para sumar scatter y randoms y alguna regularización.
	\bug
	\warning
	\author Martín Belzunce (martin.a.belzunce@gmail.com)
	\date 2014.07.11
	\version 1.1.0
*/

#ifndef _CUMLEMKERNELS_H_
#define _CUMLEMKERNELS_H_

#include <cuda.h>
#include <cuda_runtime.h>



__global__ void cuUpdatePixelValue(float *RawImage, float *FactorRawImage, float* SumAij, SizeImage size, float threshold)
{
  // Global Pixel index
  int indexPixel = blockIdx.y * (size.nPixelsX * size.nPixelsY) + blockIdx.x * blockDim.x + threadIdx.x;
  // Memory is in contiguous address, so it can be coalesce while accessing memory.
  // Each block should have the size of a 2D Image
  if(SumAij[indexPixel] >= threshold)
    RawImage[indexPixel] = RawImage[indexPixel] * FactorRawImage[indexPixel] / SumAij[indexPixel];
  else
    RawImage[indexPixel] = 0;
}

__global__ void cuGetLikelihoodValue (float* estimated_michelogram, float* measured_michelogram, float* likelihood, int numR, int numProj, int numRings, int numSinos)
{
  int indexSino2D =  threadIdx.x + (blockIdx.x * blockDim.x);
  if(indexSino2D>=numR*numProj)
    return;
  int indiceMichelogram = indexSino2D + blockIdx.y * (numProj * numR);
  if(estimated_michelogram[indiceMichelogram]>0)
  {
    (*likelihood) += measured_michelogram[indiceMichelogram] * logf(estimated_michelogram[indiceMichelogram])
      - estimated_michelogram[indiceMichelogram];
  }
}

__global__ void cuAddVectors (float* outputInput1, float* input2, int numBinsPerSlice, int numElements)
{
  int indexSino2D =  threadIdx.x + (blockIdx.x * blockDim.x);
  if(indexSino2D>=numBinsPerSlice)
    return;
  int indiceMichelogram = indexSino2D + blockIdx.y * (numBinsPerSlice);
  if(indiceMichelogram >= numElements)
    return;
  outputInput1[indiceMichelogram] = outputInput1[indiceMichelogram] + input2[indiceMichelogram];
  if(outputInput1[indiceMichelogram] < 0)
  {
    outputInput1[indiceMichelogram] = 0;
  }
}

#endif