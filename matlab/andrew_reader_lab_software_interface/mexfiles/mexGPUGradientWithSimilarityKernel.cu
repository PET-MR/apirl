/* Kernel that computes the gradient of an image, being the gradient
 * the difference between the neighbour pixels and the central pixel
 * of a cluster.
 */
#include "mex.h"
#include "matrix.h"
#include "gpu/mxGPUArray.h"
#include <cuda.h>
#include "helper_cuda.h"
#define MAX_NUMBER_VOXELS_BOWSHER 100

enum TypeOfLocalDifference
{
	LinearSum = 1,
	Magnitud = 2
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
texture<float, 3, cudaReadModeElementType> texSimilarityImage;

// Inserts a new value (without allocating memory) in an array for the similarity weight that has already been sorted from
// greater to lower. The number is inserted in the correct position. When a new element is larger that the first element, then is removed
// and its also removed from the output values
// Return EXIT_SUCCESS if could introduce the number or EXIT_FAILURE if all the numbers were greater.
__device__ int d_InsertNumberSorted(float* array, float value, float* secondaryArray, float secondaryValue, int* numElements, int numTotalElements) // numElements:actual values in the array, numTotalValues: total number of values to be filled in the array.
{
	int i; float aux;
	if ((*numElements) == 0)
	{
		array[0] = value;
		secondaryArray[0] = secondaryValue;
		(*numElements)++;
		return EXIT_SUCCESS;
	}
	if ((*numElements) == numTotalElements)
	{
		if (value > array[0])
		{
			// Replace the first value
			array[0] = value;
			secondaryArray[0] = secondaryValue;
			// And now place it in the correct position:
			for (i = 0; i < (*numElements)-1; i ++) // The last element filled is array[(*numElements)-1]
			{
				if(array[i]>array[i+1])
				{
					// The first array:
					aux = array[i+1];
					array[i+1] = array[i];
					array[i] = aux;
					// Replicate in the second array:
					aux = secondaryArray[i+1];
					secondaryArray[i+1] = secondaryArray[i];
					secondaryArray[i] = aux;
				}
				else
					return EXIT_SUCCESS; // Because it's already sorted, so if the first is bigger the enxt one too.
			}
		}
		else
		{
			// nothing to add:
			return EXIT_FAILURE;
		}
	}
	else
	{
		// The list is incomplete so I need to add 
		// Add at the end of the list:
		array[*numElements] = value; // add a new element.
		secondaryArray[*numElements] = secondaryValue; // add a new element.
		(*numElements)++;
		// Now sort it
		for (i = (*numElements)-1; i > 0; i --)
		{
			if(array[i] < array[i-1])
			{
				aux = array[i-1];
				array[i-1] = array[i];
				array[i] = aux;
				// Replicate in the second array:
				aux = secondaryArray[i-1];
				secondaryArray[i-1] = secondaryArray[i];
				secondaryArray[i] = aux;
			}
			else
				return EXIT_SUCCESS; // Because it's already sorted, so if the first is bigger the enxt one too.
		}
	}
	return EXIT_SUCCESS; 
}

__global__ void d_LocalDifferencesWithBowsher(float *ptrGradientImage, int Nx, int Ny, int Nz, int Kx, int Ky, int Kz, int bSpatialWeight, TypeOfLocalDifference typeDiff, int numBowsherVoxels) // Nx: x for texture memory, cols in matlab matrix
{
  int i, j, k, linearIndex, numElements = 0;
  float output = 0, diffSimilarity = 0, voxelValue = 0, voxelValueSimilarityImage = 0, spatialWeight = 0, spatialWeightNorm = 0;
	float bowsherValues[MAX_NUMBER_VOXELS_BOWSHER];// Could be dynamic memory, but I'm not sure if will be for local memory
	float outputValues[MAX_NUMBER_VOXELS_BOWSHER];
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
	ptrGradientImage[linearIndex] = 0;
  voxelValue = tex3D(texImage, x+0.5f, y+0.5f, z+0.5f);
	voxelValueSimilarityImage = tex3D(texSimilarityImage, x+0.5f, y+0.5f, z+0.5f);
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
						// I could pre compute it to avoid computing it for each thread:
						spatialWeightNorm += spatialWeight;
						if (spatialWeight != 0)
							spatialWeight = (1/spatialWeight);
						
					}
					else
					{
						spatialWeight = 1; // If not spatial weight: 1
						spatialWeightNorm = 1;
					}
					diffSimilarity = fabsf(tex3D(texSimilarityImage, x-Kradius_x+i+0.5f, y-Kradius_y+j+0.5f, z-Kradius_z+k+0.5f)-voxelValueSimilarityImage);
					switch(typeDiff)
					{
						case LinearSum:
							output = spatialWeight*(tex3D(texImage, x-Kradius_x+i+0.5f, y-Kradius_y+j+0.5f, z-Kradius_z+k+0.5f)-voxelValue);
							break;
						case Magnitud:
							output = (tex3D(texImage, x-Kradius_x+i+0.5f, y-Kradius_y+j+0.5f, z-Kradius_z+k+0.5f)-voxelValue)*(tex3D(texImage, x-Kradius_x+i+0.5f, y-Kradius_y+j+0.5f, z-Kradius_z+k+0.5f)-voxelValue);
							break;
					}
					if((x==71)&&(y==83)&&(z==59))
						printf("%d %d %d %f \n", x, y, z, output);
					d_InsertNumberSorted(bowsherValues, diffSimilarity, outputValues, output, &numElements, numBowsherVoxels);
				}
			}
    }
  }
  for(i = 0; i < numBowsherVoxels; i++)
	{
		ptrGradientImage[linearIndex] += outputValues[i];
			if((x==71)&&(y==83)&&(z==59))
				printf("%d BowsherWeight:%f OutputWeight:%f \n", i, bowsherValues[i], outputValues[i]);
	}
	ptrGradientImage[linearIndex] = ptrGradientImage[linearIndex]/(spatialWeightNorm*numBowsherVoxels);
}

__global__ void d_LocalDifferencesWithSimilarityWeights(float *ptrGradientImage, int Nx, int Ny, int Nz, int Kx, int Ky, int Kz, int bSpatialWeight, TypeOfLocalDifference typeDiff, TypeOfSimilarityKernel typeSimilarity) // Nx: x for texture memory, cols in matlab matrix
{
  int i, j, k, linearIndex;
  float output = 0, voxelValue = 0, voxelValueSimilarityImage = 0, spatialWeight = 0, spatialWeightNorm = 0;
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
					switch(typeDiff)
					{
						case LinearSum:
							output += spatialWeight*(tex3D(texImage, x-Kradius_x+i+0.5f, y-Kradius_y+j+0.5f, z-Kradius_z+k+0.5f)-voxelValue);
							break;
						case Magnitud:
							output += (tex3D(texImage, x-Kradius_x+i+0.5f, y-Kradius_y+j+0.5f, z-Kradius_z+k+0.5f)-voxelValue)*(tex3D(texImage, x-Kradius_x+i+0.5f, y-Kradius_y+j+0.5f, z-Kradius_z+k+0.5f)-voxelValue);
							break;
					}
				}
			}
    }
  }
  ptrGradientImage[linearIndex] = output/spatialWeightNorm;
}

/*
 * Host code, receives arrays on cpu, the memory is trasnfer to gpu here.
 * The format is: mexGPUGradient(single(Img), Kx, Ky, Kz, spatialWeight, typeOfDifference);
 */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
    /* Declare all variables.*/
    float *inputImage;
		float *similarityImage;
    cudaArray *d_inputImage;
		cudaArray *d_similarityImage;
    float *d_outputGradient;
    int Nx, Ny, Nz, Kx, Ky, Kz, enableSpatialWeight, numBowsher;
		TypeOfLocalDifference typeDiff;
		mxGPUArray* outputGradient;
    
    /* Initialize the MathWorks GPU API. */
    mxInitGPU();
    /* Throw an error if the input is not a single array. */
    if (nrhs!=8) {
        mexErrMsgIdAndTxt("mexGradient:mexGradient(inputMatrix,Kx,Ky,Kz)",
				                  "Wrong number of input parameters.");
    }
		/* Check if the matrix is single: */
		if (!mxIsSingle(prhs[0]) || !mxIsSingle(prhs[1])) //(mxGetClassID(prhs[0]) != mxSINGLE_CLASS)
		{
			 mexErrMsgIdAndTxt("mexGradient:inputMatrix",
				                  "The input Matrix must be Single.");
		}
    inputImage = (float*) mxGetData(prhs[0]);
		similarityImage = (float*) mxGetData(prhs[1]);
    const size_t* sizeDimsArray = mxGetDimensions(prhs[0]);
		const size_t* sizeDimsArraySimilarity = mxGetDimensions(prhs[1]);
		Nx = sizeDimsArray[0]; Ny = sizeDimsArray[1]; Nz = sizeDimsArray[2]; // y:rows; x:cols; z:z
		if((Nx!=sizeDimsArraySimilarity[0])||(Ny!=sizeDimsArraySimilarity[1])||(Nz!=sizeDimsArraySimilarity[2]))
			mexErrMsgIdAndTxt("mexGradient:similarityImage",
				                  "The similarity image has a different size to the input image.");
		/* Check the other parameters: make sure the first input argument is scalar */
		if( !mxIsDouble(prhs[7]) || (mxGetNumberOfElements(prhs[7])!=1) || !mxIsDouble(prhs[2]) || (mxGetNumberOfElements(prhs[2])!=1) || !mxIsDouble(prhs[3]) || (mxGetNumberOfElements(prhs[3])!=1) ||
			!mxIsDouble(prhs[4]) || (mxGetNumberOfElements(prhs[4])!=1) || !mxIsDouble(prhs[5]) || (mxGetNumberOfElements(prhs[5])!=1) || !mxIsDouble(prhs[6]) || (mxGetNumberOfElements(prhs[6])!=1))
		{
				mexErrMsgIdAndTxt("mexGradient:Kx,Ky,Kz:notScalar",
				                  "The kernel sizes must be a scalar.");
		}
    // Load the dimensions of the images and kernel:
 		Kx = mxGetScalar(prhs[3]); Ky = mxGetScalar(prhs[2]); Kz = mxGetScalar(prhs[4]);
		enableSpatialWeight = mxGetScalar(prhs[5]); typeDiff = (TypeOfLocalDifference) mxGetScalar(prhs[6]); numBowsher = mxGetScalar(prhs[7]);
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

		// Repeat the same for the similarity image:
		checkCudaErrors(cudaMalloc3DArray(&d_similarityImage, &floatTex, extentImageSize));
		copyParams.srcPtr   = make_cudaPitchedPtr(similarityImage, extentImageSize.width*sizeof(float), extentImageSize.width, extentImageSize.height);
    copyParams.dstArray = d_similarityImage;
    copyParams.extent   = extentImageSize;
    copyParams.kind     = cudaMemcpyHostToDevice;
    checkCudaErrors(cudaMemcpy3D(&copyParams));
		// set texture parameters
    texSimilarityImage.normalized = false;                      // access with normalized texture coordinates
    texSimilarityImage.filterMode = cudaFilterModeLinear;      // linear interpolation
    texSimilarityImage.addressMode[0] = cudaAddressModeBorder;   // set to zerto the borders.
    texSimilarityImage.addressMode[1] = cudaAddressModeBorder;
    texSimilarityImage.addressMode[2] = cudaAddressModeBorder;
    // bind array to 3D texture
    checkCudaErrors(cudaBindTextureToArray(texSimilarityImage, d_similarityImage, floatTex));
		
    /*
     * Call the kernel using the CUDA runtime API. We are using a 1-d grid here,
     * and it would be possible for the number of elements to be too large for
     * the grid. For this example we are not guarding against this possibility.
     */
		dim3 threadsPerBlock = dim3(8,8,8);
		dim3 blocksPerGrid = dim3(ceil((float)Nx/8),ceil((float)Ny/8),ceil((float)Nz/8));
		mexPrintf("%d %d %d %d %d %d, blocks: %d %d %d, grid: %d %d %d", Nx, Ny, Nz, Kx, Ky, Kz, threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z, blocksPerGrid.x, blocksPerGrid.y, blocksPerGrid.z);
		d_LocalDifferencesWithBowsher<<<blocksPerGrid, threadsPerBlock>>>(d_outputGradient, Nx, Ny, Nz, Kx, Ky, Kz, enableSpatialWeight, typeDiff, numBowsher);
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
