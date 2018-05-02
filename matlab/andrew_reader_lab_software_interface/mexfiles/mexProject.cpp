/* Kernel that computes the gradient of an image, being the gradient
 * the difference between the neighbour pixels and the central pixel
 * of a cluster.
 */
#include "mex.h"
#include "matrix.h"
#include <iostream>
#include <string.h>
#include <Sinogram3DSiemensMmr.h>
#include <CuProjector.h>
#include <CuProjectorInterface.h>
#include <readCudaParameters.h>
#include <Projector.h>
#include <SiddonProjector.h>
#include <Images.h>

/*
 *Receives an array on cpu with an image and then a set of parameters for the projector and the data.
 * The format is: mexProject(single(image), structImageSize, structSinogramSize, strSinogramType, structProjector, subsetIndex, numberOfSubsets);
 */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
    /* Declare all variables.*/
    float *inputImagePtr;
    float *outputSinogram;
    char* strOutputType;
    int strlen;
    /* Parameters for image and sinograms */
    double *imageSize_voxels, *voxelSize_mm;
    SizeImage mySizeImage;
    int numberOfSubsets, subsetIndex;
    float radioFov_mm = 0, axialFov_mm = 0, radioScanner_mm = 0;
    int numProj, numR, numRings, numSegments;
    int *numSinogramsPerSegment, *minRingDiffPerSegment, *maxRingDiffPerSegment;
    /* Parameters for the projector */
    int numSamples, numAxialSamples;
    char* strProjectorType;
    /* APIRL variables */
    Image* inputImage;
	Projector* forwardprojector = NULL;
    
    /* Throw an error if the input is not a single array. */
    if (nrhs<5) 
    {
        mexErrMsgIdAndTxt("mexProject:mexProject(image, structImageSize, structSinogramSize, strScanner, structProjector)",
				                  "Wrong number of input parameters.");
    }
    else if (nrhs==5) 
    {
        // no subsets
        numberOfSubsets = 0; subsetIndex = 0;
    }
    else if (nrhs!=7)
    {
        // For subsets:
        mexErrMsgIdAndTxt("mexProject:mexProject(image, structImageSize, structSinogramSize, strScanner, structProjector, subsetIndex, numberOfSubsets)",
				                  "Wrong number of input parameters.");
    }
    else
    {
        // Get subsets index and number:
        subsetIndex = mxGetScalar(prhs[5]); numberOfSubsets = mxGetScalar(prhs[6]);
    }
    /* Check if the input image is single and load the image */
    if (!mxIsSingle(prhs[0])) //(mxGetClassID(prhs[0]) != mxSINGLE_CLASS)
    {
        mexErrMsgIdAndTxt("mexProject:inputMatrix", "The input Matrix must be Single.");
    }
    inputImagePtr = (float*) mxGetData(prhs[0]);
    const size_t* sizeDimsArray = mxGetDimensions(prhs[0]);
    
    /* Check image size with structure: */
    if (!mxIsStruct(prhs[1]))
    {
        mexErrMsgIdAndTxt("mexProject:structImageSize", "The second parameter needs to be an image_size structure.");
    }
    imageSize_voxels = (double*)(mxGetPr(mxGetField(prhs[1],0,"matrixSize")));
    voxelSize_mm = (double*)mxGetPr(mxGetField(prhs[1],0,"voxelSize_mm"));
    if ((sizeDimsArray[0] != imageSize_voxels[0]) || (sizeDimsArray[1] != imageSize_voxels[1]) || (sizeDimsArray[2] != imageSize_voxels[2]))
         mexErrMsgIdAndTxt("mexProject:structImageSize", "The size of the input image is different to the size of the image_size structure.");
    // Initialize structure:
	mySizeImage.nDimensions = 3; mySizeImage.nPixelsX = imageSize_voxels[0]; mySizeImage.nPixelsY = imageSize_voxels[1]; mySizeImage.nPixelsZ = imageSize_voxels[2]; 
    mySizeImage.sizePixelX_mm = voxelSize_mm[0]; mySizeImage.sizePixelY_mm = voxelSize_mm[1]; mySizeImage.sizePixelZ_mm = voxelSize_mm[2]; 
	/* Creat input image: */
    inputImage = new Image(mySizeImage);
    memcpy(inputImage->getPixelsPtr(), inputImagePtr, inputImage->getPixelCount()*sizeof(float));
    // mexPrintf("%d %d %d %d %f %f\n", mySizeImage.nPixelsX, mySizeImage.nPixelsY, mySizeImage.nPixelsZ, inputImage->getPixelCount(), inputImage->getPixelValue(2, 10, 45), inputImagePtr[344*344*45+10*344+2]);
    
    /* Check sinogram size with structure: */
    if (!mxIsStruct(prhs[2]))
    {
        mexErrMsgIdAndTxt("mexProject:structSinogramSize", "The third parameter needs to be a singram_size structure.");
    }
    numProj = mxGetScalar(mxGetField(prhs[2],0,"nAnglesBins")); numR = mxGetScalar(mxGetField(prhs[2],0,"nRadialBins")); numRings = mxGetScalar(mxGetField(prhs[2],0,"nRings")); 
    numSegments = mxGetScalar(mxGetField(prhs[2],0,"nSeg")); numSinogramsPerSegment = (int*)(mxGetData(mxGetField(prhs[2],0,"nPlanesPerSeg"))); 
    minRingDiffPerSegment  = (int*)(mxGetData(mxGetField(prhs[2],0,"minRingDiffs"))); maxRingDiffPerSegment = (int*)(mxGetData(mxGetField(prhs[2],0,"maxRingDiffs")));
    
    // Get sinogram type:
    /* Get number of characters in the input string.  Allocate enough
       memory to hold the converted string. */
    strlen = mxGetN(prhs[3]) + 1;
    strOutputType = (char*)mxMalloc(strlen);
    /* Copy the string data into buf. */ 
    mxGetString(prhs[3], strOutputType, (mwSize)strlen); 
    
    // Get projector type
    strlen = mxGetN(mxGetField(prhs[4],0,"type")) + 1;
    strProjectorType = (char*)mxMalloc(strlen);
    /* Copy the string data into buf. */ 
    mxGetString(mxGetField(prhs[4],0,"type"), strProjectorType, (mwSize)strlen); 
    
    // Get numSamples:
    numSamples = mxGetScalar(mxGetField(prhs[4],0,"nRays")); numAxialSamples = mxGetScalar(mxGetField(prhs[4],0,"nAxialRays"));
    
    // Initialize projector parameters:
	if(strcmp(strProjectorType, "Siddon") == 0)
	{
	  forwardprojector = (Projector*)new SiddonProjector(numSamples, numAxialSamples);
	}
    else if(strcmp(strProjectorType, "CuSiddonProjector") == 0)
    {
        int gpuId; double* auxBlockSize; dim3 projectorBlockSize;
        CuProjector* cuProjector;
        CuProjectorInterface* cuProjectorInterface;
        gpuId = mxGetScalar(mxGetField(prhs[4],0,"gpuId")); auxBlockSize = (double*)mxGetPr(mxGetField(prhs[4],0,"blockSize"));
        projectorBlockSize.x = auxBlockSize[0];  projectorBlockSize.y = auxBlockSize[1];  projectorBlockSize.z = auxBlockSize[2];

        cuProjector = (CuProjector*)new CuSiddonProjector(numSamples, numAxialSamples);
        
        cuProjectorInterface = new CuProjectorInterface(cuProjector);
        // Set the configuration:
        cuProjectorInterface->setGpuId(gpuId);
        cuProjectorInterface->setProjectorBlockSizeConfig(projectorBlockSize);
        forwardprojector = (Projector*)cuProjectorInterface;
        
    }
    // Initialize input data:
    // Variables needed: rFov_mm, &axialFov_mm, &rScanner_mm, outputType
    Sinogram3D* outputProjection;
    if(strcmp(strOutputType, "cylindrical")==0)
	{
	  outputProjection = new Sinogram3DCylindricalPet(numProj, numR, numRings, radioFov_mm, axialFov_mm, radioScanner_mm, 
                                                      numSegments, numSinogramsPerSegment, minRingDiffPerSegment, maxRingDiffPerSegment);
    }
    else if(strcmp(strOutputType, "mMR")==0)
	{
        // Sinograma 3D        
        outputProjection = new Sinogram3DSiemensMmr(numProj, numR, numRings, radioFov_mm, axialFov_mm, radioScanner_mm, 
                                                    numSegments, numSinogramsPerSegment, minRingDiffPerSegment, maxRingDiffPerSegment);
        outputProjection = new Sinogram3DSiemensMmr((char*)"test.h33");
	}
	if (numberOfSubsets != 0)
        outputProjection = outputProjection->getSubset(subsetIndex, numberOfSubsets);
    forwardprojector->Project(inputImage, outputProjection);
	  
	  

    /* Return array */
    /* Wrap the result up as a MATLAB gpuArray for return. */
   // plhs[0] = mxGPUCreateMxArrayOnCPU(outputGradient); //mxGPUCreateMxArrayOnCPU(mxGPUArray const * const);
    size_t dims[3]; dims[0] = numR; dims[1] = numProj; dims[2] = outputProjection->getNumSinograms();

    plhs[0] = mxCreateUninitNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);
    outputProjection->copyRawDataInPtr((float*)mxGetData(plhs[0]));
    
    return;
}
