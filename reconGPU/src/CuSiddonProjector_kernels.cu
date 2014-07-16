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
#include <cuda.h>
#include <cuda_runtime.h>


// Memoria constante con los valores de los angulos de la proyeccion,
extern float cuda_values_theta[MAX_PHI_VALUES];

// Memoria constante con los valores de la distancia r.
extern float cuda_values_r[MAX_R_VALUES];

// Memoria constante con los valores de la coordenada axial o z.
extern float cuda_values_z[MAX_Z_VALUES];

// Memoria constante con los valores de la coordenada axial o z.
extern float cuda_values_z[MAX_Z_VALUES*MAX_Z_VALUES][MAX_SPAN];
	
__global__ void CUDA_Forward_Projection (float* volume, float* michelogram, float* michelogram_measured, int numR, int numProj, int numRings)
{
  int indexSino2D =  threadIdx.x + (blockIdx.x * cuda_threads_per_block);
  if(indexSino2D>=cudaBinsSino2D)
    return;
  int iR = indexSino2D % cuda_michelogram_size.NR;
  int iProj = (int)((float)indexSino2D / (float)cuda_michelogram_size.NR);
  int iZ = blockIdx.y;	// Block index y, index of the 2d sinogram, from which it can be taken  position Z1 and Z2 of th sinogram2d
  int indexRing1 = iZ%cuda_michelogram_size.NZ; //Ring 1 : Las columnas;
  int indexRing2 = (int)(iZ/cuda_michelogram_size.NZ);	// Ring 2 las filas	
  float4 P1;// = make_float4(0,0,0);
  float4 P2;
  float4 LOR;
  int indiceMichelogram = iR + iProj * cuda_michelogram_size.NR
	  + iZ * (cuda_michelogram_size.NProj * cuda_michelogram_size.NR);
  if(michelogram_measured[indiceMichelogram] != 0)
  {
    CUDA_GetPointsFromLOR(cuda_values_phi[iProj], cuda_values_r[iR], cuda_values_z[indexRing1], cuda_values_z[indexRing2], cudaRscanner, &P1, &P2);
    LOR.x = P2.x - P1.x;
    LOR.y = P2.y - P1.y;
    LOR.z = P2.z - P1.z;
    CUDA_Siddon (&LOR, &P1, volume, michelogram, PROJECTION, indiceMichelogram);
  }	
}