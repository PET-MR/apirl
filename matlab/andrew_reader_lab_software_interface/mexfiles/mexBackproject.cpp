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
 * The format is: mexProject(single(sinogram), structSinogramSize, strSinogramType, structImageSize, structProjector, subsetIndex, numberOfSubsets);
 * For subsets, it expects the reduceed sinogram.
 */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
    /* Declare all variables.*/
    float *outputImagePtr;
    float *inputSinogramPtr;
    char* strInputType;
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
    Image* outputImage;
    Projector* backprojector = NULL;
    
    /* Throw an error if the input is not a single array. */
    if (nrhs<5) 
    {
        mexErrMsgIdAndTxt("mexBackproject:mexBackproject(sinogram, structSinogramSize, strSinogramType, structImageSize, structProjector)",
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
        mexErrMsgIdAndTxt("mexBackproject:mexBackproject(sinogram, structSinogramSize, strSinogramType, structImageSize, structProjector, subsetIndex, numberOfSubsets)",
				                  "Wrong number of input parameters.");
    }
    else
    {
        // Get subsets index and number:
        subsetIndex = mxGetScalar(prhs[5]); numberOfSubsets = mxGetScalar(prhs[6]);
    }
    /* Check if the input sinogram is single and load the data */
    if (!mxIsSingle(prhs[0])) //(mxGetClassID(prhs[0]) != mxSINGLE_CLASS)
    {
        mexErrMsgIdAndTxt("mexBackproject:inputMatrix", "The input sinogram must be Single.");
    }
    inputSinogramPtr = (float*) mxGetData(prhs[0]);
    const size_t* sizeDimsArray = mxGetDimensions(prhs[0]);
    
    /* Check sinogram size with structure: */
    if (!mxIsStruct(prhs[1]))
    {
        mexErrMsgIdAndTxt("mexBackproject:structSinogramSize", "The third parameter needs to be a singram_size structure.");
    }
    numProj = mxGetScalar(mxGetField(prhs[1],0,"nAnglesBins")); numR = mxGetScalar(mxGetField(prhs[1],0,"nRadialBins")); numRings = mxGetScalar(mxGetField(prhs[1],0,"nRings")); 
    numSegments = mxGetScalar(mxGetField(prhs[1],0,"nSeg")); numSinogramsPerSegment = (int*)(mxGetData(mxGetField(prhs[1],0,"nPlanesPerSeg"))); 
    minRingDiffPerSegment  = (int*)(mxGetData(mxGetField(prhs[1],0,"minRingDiffs"))); maxRingDiffPerSegment = (int*)(mxGetData(mxGetField(prhs[1],0,"maxRingDiffs")));
    // Get sinogram type:
    /* Get number of characters in the input string.  Allocate enough
       memory to hold the converted string. */
    strlen = mxGetN(prhs[2]) + 1;
    strInputType = (char*)mxMalloc(strlen);
    /* Copy the string data into buf. */ 
    mxGetString(prhs[2], strInputType, (mwSize)strlen); 
    // Init sinogram object and copy data:
    Sinogram3D* inputProjection;
    if(strcmp(strInputType, "cylindrical")==0)
    {
        inputProjection = new Sinogram3DCylindricalPet(numProj, numR, numRings, radioFov_mm, axialFov_mm, radioScanner_mm, 
                                                      numSegments, numSinogramsPerSegment, minRingDiffPerSegment, maxRingDiffPerSegment);
    }
    else if(strcmp(strInputType, "mMR")==0)
    {
        // Sinograma 3D        
        inputProjection = new Sinogram3DSiemensMmr(numProj, numR, numRings, numSegments, numSinogramsPerSegment, minRingDiffPerSegment, maxRingDiffPerSegment);
    }
    // If subset reduce the data before copying:
    if (numberOfSubsets != 0)
        inputProjection = inputProjection->getSubset(subsetIndex, numberOfSubsets);
    inputProjection->readRawDataFromPtr(inputSinogramPtr);
//    mexPrintf("%f %f", inputProjection->getSegment(0)->getSinogram2D(64)->getSinogramBin(170,220));
    /* Check image size with structure: */
    if (!mxIsStruct(prhs[3]))
    {
        mexErrMsgIdAndTxt("mexBackproject:structImageSize", "The second parameter needs to be an image_size structure.");
    }
    imageSize_voxels = (double*)(mxGetPr(mxGetField(prhs[3],0,"matrixSize")));
    voxelSize_mm = (double*)mxGetPr(mxGetField(prhs[3],0,"voxelSize_mm"));
    // Initialize structure:
    mySizeImage.nDimensions = 3; mySizeImage.nPixelsX = imageSize_voxels[0]; mySizeImage.nPixelsY = imageSize_voxels[1]; mySizeImage.nPixelsZ = imageSize_voxels[2]; 
    mySizeImage.sizePixelX_mm = voxelSize_mm[0]; mySizeImage.sizePixelY_mm = voxelSize_mm[1]; mySizeImage.sizePixelZ_mm = voxelSize_mm[2]; 
    /* Creat output image: */
    outputImage = new Image(mySizeImage);    
    
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
        backprojector = (Projector*)new SiddonProjector(numSamples, numAxialSamples);
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
        backprojector = (Projector*)cuProjectorInterface;
        
    }

    // Backproject:
    backprojector->Backproject(inputProjection, outputImage);

    /* Return array */
    /* Wrap the result up as a MATLAB gpuArray for return. */
   // plhs[0] = mxGPUCreateMxArrayOnCPU(outputGradient); //mxGPUCreateMxArrayOnCPU(mxGPUArray const * const);
    size_t dims[3]; dims[0] = mySizeImage.nPixelsX; dims[1] = mySizeImage.nPixelsY; dims[2] = mySizeImage.nPixelsZ;
    plhs[0] = mxCreateUninitNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);
    memcpy(mxGetData(plhs[0]), outputImage->getPixelsPtr(), outputImage->getPixelCount()*sizeof(float));
    
    return;
}
