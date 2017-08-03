/* Kernel that computes the gradient of an image, being the gradient
 * the difference between the neighbour pixels and the central pixel
 * of a cluster.
 */
#include "mex.h"
#include "matrix.h"
#include "gpu/mxGPUArray.h"
#include <cuda.h>
#include "helper_cuda.h"

/* Texture memory */
texture<float, 3, cudaReadModeElementType> texImage;

__global__ void d_Gradient(float *ptrGradientImage, int Nx, int Ny, int Nz, int Kx, int Ky, int Kz) // Nx: x for texture memory, cols in matlab matrix
{
  int i, j, k, linearIndex;
  float output = 0, voxelValue = 0;
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
				//if((y<(Ny-Kradius_y))||(x<(Nx-Kradius_x))||(z<(Nz-Kradius_z))|(y>Kradius_y)||(x>Kradius_x)||(z>Kradius_z))
					// Sum of differences
					output += tex3D(texImage, x-Kradius_x+i+0.5f, y-Kradius_y+j+0.5f, z-Kradius_z+k+0.5f)-voxelValue;
			}
    }
  }
  ptrGradientImage[linearIndex] = output;
}

__global__ void d_GradientTV(float *ptrGradientImage, int Nx, int Ny, int Nz, int Kx, int Ky, int Kz) // Its computed: sqrt(sum(delta_ij^2))
{
  int i, j, k, linearIndex;
  float output = 0, voxelValue = 0;
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
				//if((y<(Ny-Kradius_y))||(x<(Nx-Kradius_x))||(z<(Nz-Kradius_z))|(y>Kradius_y)||(x>Kradius_x)||(z>Kradius_z))
					// Sum of differences
					output += (tex3D(texImage, x-Kradius_x+i+0.5f, y-Kradius_y+j+0.5f, z-Kradius_z+k+0.5f)-voxelValue).*(tex3D(texImage, x-Kradius_x+i+0.5f, y-Kradius_y+j+0.5f, z-Kradius_z+k+0.5f)-voxelValue);
			}
    }
  }
  ptrGradientImage[linearIndex] = sqrt(output);
}

/*
 * Host code, receives arrays on cpu, the memory is trasnfer to gpu here.
 */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
    /* Declare all variables.*/
    float *inputImage;
    cudaArray *d_inputImage;
    float *d_outputGradient;
    int Nx, Ny, Nz, Kx, Ky, Kz;
		mxGPUArray* outputGradient;
    
    /* Initialize the MathWorks GPU API. */
    mxInitGPU();

    /* Throw an error if the input is not a single array. */
    if (nrhs!=4) {
        mexErrMsgIdAndTxt("mexGradient:mexGradient(inputMatrix,Kx,Ky,Kz)",
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
		if( !mxIsDouble(prhs[1]) || (mxGetNumberOfElements(prhs[1])!=1) || !mxIsDouble(prhs[2]) || (mxGetNumberOfElements(prhs[2])!=1) || !mxIsDouble(prhs[3]) || (mxGetNumberOfElements(prhs[3])!=1)) 
		{
				mexErrMsgIdAndTxt("mexGradient:Kx,Ky,Kz:notScalar",
				                  "The kernel sizes must be a scalar.");
		}
    // Load the dimensions of the images and kernel:
 		Kx = mxGetScalar(prhs[2]); Ky = mxGetScalar(prhs[1]); Kz = mxGetScalar(prhs[3]);

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
mexPrintf("%d %d",Nx, extentImageSize.width);

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
		mexPrintf("%d %d %d %d %d %d, blocks: %d %d %d, grid: %d %d %d", Nx, Ny, Nz, Kx, Ky, Kz, threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z, blocksPerGrid.x, blocksPerGrid.y, blocksPerGrid.z);
		d_Gradient<<<blocksPerGrid, threadsPerBlock>>>(d_outputGradient, Nx, Ny, Nz, Kx, Ky, Kz);
		checkCudaErrors(cudaThreadSynchronize());

    /* Wrap the result up as a MATLAB gpuArray for return. */
    plhs[0] = mxGPUCreateMxArrayOnCPU(outputGradient); //mxGPUCreateMxArrayOnCPU(mxGPUArray const * const);

		// Undo the current texture binding so we leave things in a good state
    // for the next loop iteration or upon exiting.
    //cudaUnbindTexture(texImage);
    /*
     * The mxGPUArray pointers are host-side structures that refer to device
     * data. These must be destroyed before leaving the MEX function.
     */
    //mxGPUDestroyGPUArray(inputImage); 
    mxGPUDestroyGPUArray(outputGradient);
}
