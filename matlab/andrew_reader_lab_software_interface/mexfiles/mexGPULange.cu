/* Kernel that computes the gradient of an image, being the gradient
 * the difference between the neighbour pixels and the central pixel
 * of a cluster.
 */
#include "mex.h"
#include "matrix.h"
#include "gpu/mxGPUArray.h"
#include <cuda.h>
#include "helper_cuda.h"

enum TypeOfLange
{
	Local = 1,
	NonLocal = 2
};

enum TypeOfSimilarityKernel
{
	None = 0,
	Bowsher = 1,
	JointGaussianKernel = 2,
	JointEntropy = 3
};

/* Texture memory */
texture<float, 3, cudaReadModeElementType> texImage;

__global__ void d_Lange(float *ptrGradientImage, int Nx, int Ny, int Nz, int Kx, int Ky, int Kz, float delta, int bSpatialWeight, TypeOfLange typeLange) // Nx: x for texture memory, cols in matlab matrix
{
  int i, j, k, linearIndex;
  float output = 0, voxelValue = 0, spatialWeight = 0, spatialWeightNorm = 0, auxValue = 0, normValue = 0;
	int Kradius_x, Kradius_y, Kradius_z;
	Kradius_x = Kx/2; Kradius_y = Ky/2; Kradius_z = Kz/2;
  // The blocks are larger than the voxel to be processed to load the edges of the kernel
  int x = threadIdx.x + blockDim.x*blockIdx.x;
  int y = threadIdx.y + blockDim.y*blockIdx.y;
  int z = threadIdx.z + blockDim.z*blockIdx.z;

  // Check if inside the image
  if((y>=Ny)||(x>=Nx)||(z>=Nz)|(y<0)||(x<0)||(z<0))
  {
    return;
  }
  linearIndex = x + y*Nx + z*Nx*Ny; // col-wise stored matrix for matlab
	
  voxelValue = tex3D(texImage, x+0.5f, y+0.5f, z+0.5f);
  
  // Process only the voxels inside the processing window
  #pragma unroll
  for(i = 0; i < Kx; i++)
  {
    #pragma unroll
    for(j = 0; j < Ky; j++)
    {
			#pragma unroll
			for(k = 0; k < Kz; k++)
			{
				if((y<(Ny-Kradius_y))||(x<(Nx-Kradius_x))||(z<(Nz-Kradius_z))||(y>Kradius_y)||(x>Kradius_x)||(z>Kradius_z))
				{
					// Sum of differences
					// spatial weight:
					if (bSpatialWeight)
					{
						spatialWeight = sqrt((-Kradius_x+(float)i)*(-Kradius_x+(float)i)+(-Kradius_y+(float)j)*(-Kradius_y+(float)j)+(-Kradius_z+(float)k)*(-Kradius_z+(float)k));
						if (spatialWeight != 0)
							spatialWeight = (1/spatialWeight);
						// I could pre compute it to avoid computing it for each thread:
						spatialWeightNorm += spatialWeight;
					}
					else
					{
						spatialWeight = 1; // If not spatial weight: 1
						spatialWeightNorm = 1;
					}
                    // Delta value:
                    auxValue = tex3D(texImage, x-Kradius_x+i+0.5f, y-Kradius_y+j+0.5f, z-Kradius_z+k+0.5f)-voxelValue;
					switch(typeLange)
					{
						case Local:
							output += spatialWeight*auxValue/(delta + fabs(auxValue));
							break;
						case NonLocal:
							output += spatialWeight*auxValue;
                            normValue += spatialWeight*auxValue*auxValue;
							break;
					}
				}
			}
    }
  }
  switch(typeLange)
	{
		case Local:
			ptrGradientImage[linearIndex] = -output/spatialWeightNorm;
			break;
		case NonLocal:
			ptrGradientImage[linearIndex] = -(output/spatialWeightNorm)/(delta+sqrt(normValue/spatialWeightNorm));
			break;
	}
}

/*
 * Host code, receives arrays on cpu, the memory is trasnfer to gpu here.
 * The format is: mexGPUGradient(single(Img), Kx, Ky, Kz, delta, spatialWeight, typeOfLange);
 */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
    /* Declare all variables.*/
    float delta;
    float *inputImage;
    cudaArray *d_inputImage;
    float *d_outputGradient;
    int Nx, Ny, Nz, Kx, Ky, Kz, enableSpatialWeight;
		TypeOfLange typeLange;
		mxGPUArray* outputGradient;
    
    /* Initialize the MathWorks GPU API. */
    mxInitGPU();

    /* Throw an error if the input is not a single array. */
    if (nrhs!=7) {
        mexErrMsgIdAndTxt("mexGradient:mexGradient(inputMatrix,Kx,Ky,Kz,delta,spatialWeight,type)",
				                  "Wrong number of input parameters.");
    }
		/* Check if the matrix is single: */
		if (!mxIsSingle(prhs[0])) //(mxGetClassID(prhs[0]) != mxSINGLE_CLASS)
		{
			 mexErrMsgIdAndTxt("mexGradient:inputMatrix",
				                  "The input Matrix must be Single.");
		}
    inputImage = (float*) mxGetData(prhs[0]);
    const size_t* sizeDimsArray = mxGetDimensions(prhs[0]);
		Nx = sizeDimsArray[0]; Ny = sizeDimsArray[1]; Nz = sizeDimsArray[2]; // y:rows; x:cols; z:z
		
		/* Check the other parameters: make sure the first input argument is scalar */
		if( !mxIsDouble(prhs[1]) || (mxGetNumberOfElements(prhs[1])!=1) || !mxIsDouble(prhs[2]) || (mxGetNumberOfElements(prhs[2])!=1) || !mxIsDouble(prhs[3]) || (mxGetNumberOfElements(prhs[3])!=1) ||
			!mxIsDouble(prhs[4]) || (mxGetNumberOfElements(prhs[4])!=1) || !mxIsDouble(prhs[5]) || (mxGetNumberOfElements(prhs[5])!=1) || !mxIsDouble(prhs[6]) || (mxGetNumberOfElements(prhs[6])!=1))
		{
				mexErrMsgIdAndTxt("mexGradient:Kx,Ky,Kz,delta,spatialWeight,type:notScalar",
				                  "The kernel sizes must be a scalar.");
		}
    // Load the dimensions of the images and kernel:
 		Kx = mxGetScalar(prhs[2]); Ky = mxGetScalar(prhs[1]); Kz = mxGetScalar(prhs[3]);
		delta = mxGetScalar(prhs[4]); enableSpatialWeight = mxGetScalar(prhs[5]); typeLange = (TypeOfLange) mxGetScalar(prhs[6]);
    /* Create a GPUArray to hold the result and get its underlying pointer. mxGetClassID(prhs[0])*/
    outputGradient = mxGPUCreateGPUArray(mxGetNumberOfDimensions(prhs[0]),
                            mxGetDimensions(prhs[0]),
                            mxSINGLE_CLASS,
                            mxREAL,
                            MX_GPU_INITIALIZE_VALUES);
    d_outputGradient = (float *)(mxGPUGetData(outputGradient));

		//checkCudaErrors(cudaMalloc((void**) &d_outputGradient, sizeof(float)*Nx*Ny*Nz));
		// Allocating memory for texture, initializing and binding:
		cudaChannelFormatDesc floatTex = cudaCreateChannelDesc<float>();
		const cudaExtent extentImageSize = make_cudaExtent(Nx, Ny, Nz); // For this memory the width is Nx, therefore the cols
		cudaMemcpy3DParms copyParams = {0};

	  // Must be called with init gpu memory. It loads the texture memory for the projection.
    // The image is in a texture memory:  cudaChannelFormatDesc floatTex;
    checkCudaErrors(cudaMalloc3DArray(&d_inputImage, &floatTex, extentImageSize));
		// Copy params
		copyParams.srcPtr   = make_cudaPitchedPtr(inputImage, extentImageSize.width*sizeof(float), extentImageSize.width, extentImageSize.height);
    copyParams.dstArray = d_inputImage;
    copyParams.extent   = extentImageSize;
    copyParams.kind     = cudaMemcpyHostToDevice;
    checkCudaErrors(cudaMemcpy3D(&copyParams));
    // set texture parameters
    texImage.normalized = false;                      // access with normalized texture coordinates
    texImage.filterMode = cudaFilterModeLinear;      // linear interpolation
    texImage.addressMode[0] = cudaAddressModeBorder;   // set to zerto the borders.
    texImage.addressMode[1] = cudaAddressModeBorder;
    texImage.addressMode[2] = cudaAddressModeBorder;
    // bind array to 3D texture
    checkCudaErrors(cudaBindTextureToArray(texImage, d_inputImage, floatTex));

    /*
     * Call the kernel using the CUDA runtime API. We are using a 1-d grid here,
     * and it would be possible for the number of elements to be too large for
     * the grid. For this example we are not guarding against this possibility.
     */
		dim3 threadsPerBlock = dim3(8,8,8);
		dim3 blocksPerGrid = dim3(ceil((float)Nx/8),ceil((float)Ny/8),ceil((float)Nz/8));
		//mexPrintf("%d %d %d %d %d %d, blocks: %d %d %d, grid: %d %d %d", Nx, Ny, Nz, Kx, Ky, Kz, threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z, blocksPerGrid.x, blocksPerGrid.y, blocksPerGrid.z);
		d_Lange<<<blocksPerGrid, threadsPerBlock>>>(d_outputGradient, Nx, Ny, Nz, Kx, Ky, Kz, delta, enableSpatialWeight, typeLange);
		checkCudaErrors(cudaThreadSynchronize());

    /* Wrap the result up as a MATLAB gpuArray for return. */
    plhs[0] = mxGPUCreateMxArrayOnCPU(outputGradient); //mxGPUCreateMxArrayOnCPU(mxGPUArray const * const);

		// Undo the current texture binding so we leave things in a good state
    // for the next loop iteration or upon exiting.
    cudaUnbindTexture(texImage);
    /*
     * The mxGPUArray pointers are host-side structures that refer to device
     * data. These must be destroyed before leaving the MEX function.
     */
    //mxGPUDestroyGPUArray(inputImage); 
    mxGPUDestroyGPUArray(outputGradient);
    // Free the other memory:
		cudaFree(d_outputGradient);
		cudaFreeArray(d_inputImage);
}
