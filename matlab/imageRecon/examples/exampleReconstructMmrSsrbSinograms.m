%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 27/01/2015
%  *********************************************************************
%  This example reads an span 1 sinogram of the mMr scanner, reads an
%  attenuation map, reads a normalization file and then gets the rebinned
%  sinograms using ssrb, applies the corrections and then performs a 2d
%  reconstruction. For the attenuation correction, the acf factors are
%  obtained through APIRL library.

clear all 
close all
%% PATHS FOR EXTERNAL FUNCTIONS AND RESULTS
addpath('/workspaces/Martin/KCL/Biograph_mMr/mmr');
apirlPath = '/workspaces/Martin/PET/apirl-code/trunk/';
addpath(genpath([apirlPath '/matlab']));
setenv('PATH', [getenv('PATH') ':/workspaces/Martin/PET/apirl-code/trunk/build/debug/bin']);
setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') ':/workspaces/Martin/PET/apirl-code/trunk/build/debug/bin']);
outputPath = '/workspaces/Martin/KCL/Biograph_mMr/mmr/5hr_ge68/';
%setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') ':/usr/lib/x86_64-linux-gnu/']);
%% READING THE SINOGRAMS
% Read the sinograms:
%filenameUncompressedMmr = '/workspaces/Martin/KCL/Biograph_mMr/mmr/test.s';
%outFilenameIntfSinograms = '/workspaces/Martin/KCL/Biograph_mMr/mmr/testIntf';
filenameUncompressedMmr = '/workspaces/Martin/KCL/Biograph_mMr/mmr/5hr_ge68/cylinder_5hours.s';
outFilenameIntfSinograms = '/workspaces/Martin/KCL/Biograph_mMr/mmr/5hr_ge68/cylinder_5hoursIntf';
[sinogram, delayedSinogram, structSizeSino3d] = getIntfSinogramsFromUncompressedMmr(filenameUncompressedMmr, outFilenameIntfSinograms);

% Read the normalization factors:
filenameRawData = '/workspaces/Martin/KCL/Biograph_mMr/mmr/Norm_20141008101010.n';
[componentFactors, componentLabels]  = readmMrComponentBasedNormalization(filenameRawData, 1);
%% NORMALIZATION FACTORS
% A set of 2d sinograms with normalization effects is
% Geometric Normalization:
normSinoGeom = 1./ imresize(double(componentFactors{1}'), [structSizeSino3d.numTheta structSizeSino3d.numR]);
figure;
imshow((1./normSinoGeom) ./max(max((1./normSinoGeom))));
% Crystal interference, its a pattern that is repeated peridoically:
crystalInterf = double(componentFactors{2});
crystalInterf = 1./repmat(crystalInterf,structSizeSino3d.numTheta/size(crystalInterf,1),1);
% Geometric Normalization and crystal interference:
figure;
imshow(crystalInterf ./max(max(crystalInterf)));
% Generate one norm factor:
normSinoGeomInterf = normSinoGeom .* crystalInterf;
title('Sinogram Correction For Crystal Interference and Geometry');
%% REBINNING SINOGRAMS
% Create a rebinned 2d sinograms from the complete 3d sinogram:
[sinograms2d, structSizeSino2d]  = ssrbFromSinogram3d(sinogram, structSizeSino3d);
interfileWriteSino(single(sinograms2d), '/workspaces/Martin/KCL/Biograph_mMr/mmr/5hr_ge68/ssrb2dSinograms');

% Create ACFs of a computed phatoms with the linear attenuation
% coefficients:
acfFilename = 'acfsSinograms2dSsrb';
filenameSinogram = '/workspaces/Martin/KCL/Biograph_mMr/mmr/5hr_ge68/ssrb2dSinograms';
acfs = createNormalizationPhantomACFs(outputPath, acfFilename, filenameSinogram, structSizeSino2d, 1);

% Correct the sinograms for normalization and the write it in interfile:
correctedSinogram = zeros(structSizeSino2d.numTheta, structSizeSino2d.numR, structSizeSino2d.numZ);
corrSinoFilename = [outputPath '/correctedSinograms2dSsrb'];
for i = 1 : structSizeSino2d.numZ
    correctedSinogram(:,:,i) = sinograms2d(:,:,i) .* normSinoGeomInterf;
end
interfileWriteSino(correctedSinogram, corrSinoFilename);

% Create initial estimate for reconstruction:
% Size of the image to cover the full fov:
sizeImage_mm = [structSizeSino2d.rFov_mm*2 structSizeSino2d.rFov_mm*2 structSizeSino2d.zFov_mm];
% The size in pixels based in numR and the number of rings:
sizeImage_pixels = [structSizeSino2d.numR structSizeSino2d.numR structSizeSino2d.numZ];
% Size of the pixels:
sizePixel_mm = sizeImage_mm ./ sizeImage_pixels;
% Inititial estimate:
initialEstimate = ones(sizeImage_pixels, 'single');
filenameInitialEstimate = [outputPath '/initialEstimateRebinning'];
interfilewrite(initialEstimate, filenameInitialEstimate, sizePixel_mm);

% Plot an average of each projecction to check the attenuation correction:
figure;
plot([mean(sinograms2d(:,:,10),1); mean(sinograms2d(:,:,10).*acfs(:,:,10),1)]');
legend('Mean Uncorrected Projection', 'Mean Corrected Projection');
nx = floor(structSizeSino2d.numR/sqrt(2))-1;
ny = floor(structSizeSino2d.numR/sqrt(2))-1;
mm_x        = 2.0445;
for slice = 1 : structSizeSino2d.numZ
    sino2d = double(sinograms2d(:,:,slice));
    gaps_sinogram = zeros(size(sino2d));
    gaps_sinogram(sino2d>0) = 1;
    gaps_sinogram = gaps_sinogram .* normSinoGeomInterf;
    acf_sino = double(acfs(:,:,slice));
    em_recon(:,:,slice) = mlem_mmr( nx, ny, mm_x, structSizeSino2d.numR, sino2d', gaps_sinogram', acf_sino', gaps_sinogram', structSizeSino2d.thetaValues_deg, 40, 0);
end

figure;
imageSlices = getImageFromSlices(em_recon,8);
imshow(imageSlices);