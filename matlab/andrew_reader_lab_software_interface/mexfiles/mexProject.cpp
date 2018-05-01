/* Kernel that computes the gradient of an image, being the gradient
 * the difference between the neighbour pixels and the central pixel
 * of a cluster.
 */
#include "mex.h"
#include "matrix.h"
#include <iostream>
#include <Sinogram3DSiemensMmr.h>
#include <CuProjector.h>
#include <CuProjectorInterface.h>
#include <readCudaParameters.h>
#include <Projector.h>
#include <SiddonProjector.h>
#include <Images.h>

/*
 *Receives an array on cpu with an image and then a set of parameters for the projector and the data.
 * The format is: mexProject(single(image), structImageSize, structSinogramSize);
 */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
    /* Declare all variables.*/
    float *inputImagePtr;
    float *outputSinogram;
    /* Parameters for image and sinograms */
    double *imageSize_voxels, *voxelSize_mm;
    SizeImage mySizeImage;
    int Nx, Ny, Nz; 
    int numberOfSubsets, subsetIndex;
    float rFov_mm = 0, axialFov_mm = 0, rScanner_mm = 0;
    /* APIRL variables */
    Image* inputImage;
	Projector* forwardprojector = NULL;
    
    /* Throw an error if the input is not a single array. */
    if (nrhs!=3) {
        mexErrMsgIdAndTxt("mexCuProject:mexCuProject(image, structImageSize, structSinogramSize)",
				                  "Wrong number of input parameters.");
    }
    /* Check if the input image is single and load the image */
    if (!mxIsSingle(prhs[0])) //(mxGetClassID(prhs[0]) != mxSINGLE_CLASS)
    {
        mexErrMsgIdAndTxt("mexCuProject:inputMatrix", "The input Matrix must be Single.");
    }
    inputImagePtr = (float*) mxGetData(prhs[0]);
    const size_t* sizeDimsArray = mxGetDimensions(prhs[0]);
    /* Check image size with structure: */
    if (!mxIsStruct(prhs[1]))
    {
        mexErrMsgIdAndTxt("mexCuProject:structImageSize", "The second parameter needs to be an image_size structure.");
    }
    imageSize_voxels = (double*)(mxGetPr(mxGetField(prhs[1],0,"matrixSize")));
    voxelSize_mm = (double*)mxGetPr(mxGetField(prhs[1],0,"voxelSize_mm"));
    if ((sizeDimsArray[0] != imageSize_voxels[0]) || (sizeDimsArray[1] != imageSize_voxels[1]) || (sizeDimsArray[2] != imageSize_voxels[2]))
         mexErrMsgIdAndTxt("mexCuProject:structImageSize", "The size of the input image is different to the size of the image_size structure.");
    // Initialize structure:
	mySizeImage.nDimensions = 3; mySizeImage.nPixelsX = imageSize_voxels[0]; mySizeImage.nPixelsY = imageSize_voxels[1]; mySizeImage.nPixelsZ = imageSize_voxels[2]; 
    mySizeImage.sizePixelX_mm = voxelSize_mm[0]; mySizeImage.sizePixelY_mm = voxelSize_mm[1]; mySizeImage.sizePixelZ_mm = voxelSize_mm[2]; 
	/* Creat input image: */
    inputImage = new Image(mySizeImage);
    memcpy(inputImage->getPixelsPtr(), inputImagePtr, inputImage->getPixelCount()*sizeof(float));
    mexPrintf("%d %d %d %d %f %f\n", mySizeImage.nPixelsX, mySizeImage.nPixelsY, mySizeImage.nPixelsZ, inputImage->getPixelCount(), inputImage->getPixelValue(2, 10, 45), inputImagePtr[344*344*45+10*344+2]);
    
    
    /*
    // Initialize projector parameters:
    // Inicializo el proyector a utilizar:
	if(strForwardprojector.compare("Siddon") == 0)
	{
	  int numSamples, numAxialSamples;
	  if(getSiddonProjectorParameters(parameterFileName, "Projection", &numSamples, &numAxialSamples))
	    return -1; // Return when is a big error, if the fields are not fouund they are already filled with the defaults.
	  forwardprojector = (Projector*)new SiddonProjector(numSamples, numAxialSamples);
	}
	#ifdef __USE_CUDA__
	  int gpuId;	// Id de la gpu a utilizar.
	  dim3 projectorBlockSize;	// Parámetros de cuda.
	  if(strForwardprojector.compare("CuSiddonProjector") == 0)
	  {
		int numSamples, numAxialSamples;
		// CuSiddonProjector now also admits multiple rays.
		if(getSiddonProjectorParameters(parameterFileName, "Projection", &numSamples, &numAxialSamples))
			return -1; // Return when is a big error, if the fields are not fouund they are already filled with the defaults.
		cuProjector = (CuProjector*)new CuSiddonProjector(numSamples, numAxialSamples);
		
	    cuProjectorInterface = new CuProjectorInterface(cuProjector);
	    // Get size of kernels:
	    if(getProjectorBlockSize(parameterFileName, "Projection", &projectorBlockSize))
	    {
	      return -1;
	    }
	    if(getGpuId(parameterFileName, "Projection", &gpuId))
	    {
	      return -1;
	    }
	    // Set the configuration:
	    cuProjectorInterface->setGpuId(gpuId);
	    cuProjectorInterface->setProjectorBlockSizeConfig(projectorBlockSize);
	    forwardprojector = (Projector*)cuProjectorInterface;
	  }
	  */
    // Initialize input data:
    // Variables needed: rFov_mm, &axialFov_mm, &rScanner_mm, outputType
    Sinogram3D* outputProjection;
    if(outputType.compare("Sinogram3D")==0)
	{
	  outputProjection = new Sinogram3DCylindricalPet(numProj, numR, numRings, radioFov_mm, axialFov_mm, radioScanner_mm, 
                                                      int numSegments, int* numSinogramsPerSegment, int* minRingDiffPerSegment, int* maxRingDiffPerSegment);
    }
    else if(outputType.compare("Sinogram3DSiemensMmr")==0)
	{
        // Sinograma 3D
        outputProjection = new Sinogram3DSiemensMmr(numProj, numR, numRings, radioFov_mm, axialFov_mm, radioScanner_mm, 
                                                    int numSegments, int* numSinogramsPerSegment, int* minRingDiffPerSegment, int* maxRingDiffPerSegment);
	}
	if (numberOfSubsets != 0)
        outputProjection = outputProjection->getSubset(subsetIndex, numberOfSubsets);
    forwardprojector->Project(inputImage, outputProjection);
    outputProjection->writeInterfile(outputFilename);
	  
	  
    // Load the dimensions of the images and kernel:
 		Kx = mxGetScalar(prhs[2]); Ky = mxGetScalar(prhs[1]); Kz = mxGetScalar(prhs[3]);
		enableSpatialWeight = mxGetScalar(prhs[4]); typeDiff = (TypeOfLocalDifference) mxGetScalar(prhs[5]);
    /* Create a GPUArray to hold the result and get its underlying pointer. mxGetClassID(prhs[0])*/



}
