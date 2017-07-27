/* Kernel that computes the gradient of an image, being the gradient
 * the difference between the neighbour pixels and the central pixel
 * of a cluster.
 */
#include "mex.h"
#include "gpu/mxGPUArray.h"

__global__ void d_Gradient(float *ptrInputImage, float *ptrGradientImage, int Nx, int Ny, int Nz, int Kx, int Ky, int Kz)
{
  int i, j, k, linearIndex, linearIndexShared;
  // Shared memory needs to be as big as the block size
  extern __shared__ float cluster[];
  float output = 0, voxelValue = 0;
  // The blocks are larger than the voxel to be processed to load the edges of the kernel
  int x = (threadIdx.x-Kx)+(blockDim.x-2*Kx)*blockIdx.x;
  int y = (threadIdx.y-Ky)+(blockDim.y-2*Ky)*blockIdx.y;
  int z = (threadIdx.y-Kz)+(blockDim.y-2*Kz)*blockIdx.y;

  // Check if inside the image
  if((y>=Ny)||(x>=Nx)||(z>=Nz)|(y<0)||(x<0)||(z<0))
  {
    return;
  }
  linearIndex = y + x*Ny + z*Nx*Ny; // col-wise stored matrix
  linearIndexShared = threadIdx.y + threadIdx.x*blockDim.y + threadIdx.z*blockDim.y*blockDim.z;
  // Each thread loads their voxel memory
  cluster[linearIndexShared] = ptrInputImage[linearIndex];
  // Get the voxel value:
  voxelValue = cluster[linearIndexShared];
  // Synchronize 
  __syncthreads();
  
  // Process only the voxels inside the processing window
  if((threadIdx.y>=Ky)&&(threadIdx.x>=Kx)&&(threadIdx.y<(blockDim.y-Ky))&&(threadIdx.y<(blockDim.y-Ky))&&(threadIdx.x<blockDim.x-Kx)&&(threadIdx.z<blockDim.z-Kz))
  {
    #pragma unroll
    for(i = 0; i < Kx; i++)
    {
      #pragma unroll
      for(j = 0; j < Ky; j++)
      {
	#pragma unroll
	for(k = 0; k < Kz; k++)
	{
	  // Sum of differences
	  linearIndexShared = (threadIdx.y-Ky+j) + (threadIdx.x-Kx+i)*blockDim.y + (threadIdx.z-Kz+k)*blockDim.y*blockDim.z;
	  output += cluster[linearIndexShared]-voxelValue;
	}
      }
    }
    ptrGradientImage[linearIndex] = (unsigned char) (output);
  }
}

/*
 * Host code
 */
void mexGradient(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
    /* Declare all variables.*/
    mxGPUArray const *inputImage;
    mxGPUArray *outputGradient;
    float const *d_inputImage;
    float *d_outputGradient;
    int N;
    char const * const errId = "parallel:gpu:mexGPUExample:InvalidInput";
    char const * const errMsg = "Invalid input to MEX file.";
    int Nx, Ny, Nz, Kx, Ky, Kz;
    /* Choose a reasonably sized number of threads for the block. */
    dim3 threadsPerBlock = dim3(512;
    int blocksPerGrid;

    /* Initialize the MathWorks GPU API. */
    mxInitGPU();

    /* Throw an error if the input is not a GPU array. */
    if ((nrhs!=1) || !(mxIsGPUArray(prhs[0]))) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }

    inputImage = mxGPUCreateFromMxArray(prhs[0]);
    // Load the dimensions of the images and kernel:
    Nx = prhs[1]; Ny = prhs[2]; Nz = prhs[3]; Kx = prhs[4]; Ky = prhs[5]; Kz = prhs[6];
    /*
     * Verify that A really is a double array before extracting the pointer.
     */
    if (mxGPUGetClassID(inputImage) != mxDOUBLE_CLASS) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }

    /*
     * Now that we have verified the data type, extract a pointer to the input
     * data on the device.
     */
    d_inputImage = (double const *)(mxGPUGetDataReadOnly(inputImage));

    /* Create a GPUArray to hold the result and get its underlying pointer. */
    outputGradient = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(inputImage),
                            mxGPUGetDimensions(inputImage),
                            mxGPUGetClassID(inputImage),
                            mxGPUGetComplexity(inputImage),
                            MX_GPU_DO_NOT_INITIALIZE);
    d_outputGradient = (double *)(mxGPUGetData(outputGradient));

    /*
     * Call the kernel using the CUDA runtime API. We are using a 1-d grid here,
     * and it would be possible for the number of elements to be too large for
     * the grid. For this example we are not guarding against this possibility.
     */
    N = (int)(mxGPUGetNumberOfElements(A));
    blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    d_Gradient<<<blocksPerGrid, threadsPerBlock>>>(d_inputImage, d_outputGradient, Nx, Ny, Nz, Kx, Ky, Kz);

    /* Wrap the result up as a MATLAB gpuArray for return. */
    plhs[0] = mxGPUCreateMxArrayOnGPU(outputGradient);

    /*
     * The mxGPUArray pointers are host-side structures that refer to device
     * data. These must be destroyed before leaving the MEX function.
     */
    mxGPUDestroyGPUArray(inputImage);
    mxGPUDestroyGPUArray(outputGradient);
}