%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 03/02/2015
%  *********************************************************************
%  This example reads an interfile span 1 sinogram, compresses to span 11,
%  generates the normalization and attenuation factors, corrects then and
%  generate the files to reconstruct them with APIRL.
clear all 
close all
%% PATHS FOR EXTERNAL FUNCTIONS AND RESULTS
addpath('/home/mab15/workspace/KCL/Biograph_mMr/mmr');
apirlPath = '/home/mab15/workspace/apirl-code/trunk/';
addpath(genpath([apirlPath '/matlab']));
setenv('PATH', [getenv('PATH') ':' apirlPath '/build/bin']);
setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') ':' apirlPath '/build/bin']);
outputPath = '/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/NEMA_IQ_20_02_2014/Reconstruction/span11/';
mkdir(outputPath);
%setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') ':/usr/lib/x86_64-linux-gnu/']);
%% READING THE SINOGRAMS
% Read the sinograms:
sinogramsPath = '/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/NEMA_IQ_20_02_2014/';
filenameUncompressedMmr = [sinogramsPath 'PET_ACQ_194_20150220154553-0uncomp.s'];
outFilenameIntfSinograms = [sinogramsPath 'NemaIq20_02_2014_ApirlIntf.s'];
[sinogram, delayedSinogram, structSizeSino3d] = getIntfSinogramsFromUncompressedMmr(filenameUncompressedMmr, outFilenameIntfSinograms);

%% CREATE SINOGRAMS3D SPAN 11
% Create sinogram span 11:
michelogram = generateMichelogramFromSinogram3D(sinogram, structSizeSino3d);
structSizeSino3dSpan11 = getSizeSino3dFromSpan(structSizeSino3d.numR, structSizeSino3d.numTheta, structSizeSino3d.numZ, ...
    structSizeSino3d.rFov_mm, structSizeSino3d.zFov_mm, 11, structSizeSino3d.maxAbsRingDiff);
sinogramSpan11 = reduceMichelogram(michelogram, structSizeSino3dSpan11);
clear michelogram
clear sinogram
% Write to a file in interfile format:
outputSinogramName = [outputPath 'sinogramSpan11'];
interfileWriteSino(single(sinogramSpan11), outputSinogramName, structSizeSino3dSpan11);
% In int 16 also:
outputSinogramName = [outputPath 'sinogramSpan11_int16'];
interfileWriteSino(int16(sinogramSpan11), outputSinogramName, structSizeSino3dSpan11);

% The same for the delayed:
% Create sinogram span 11:
michelogram = generateMichelogramFromSinogram3D(delayedSinogram, structSizeSino3d);
delaySinogramSpan11 = reduceMichelogram(michelogram, structSizeSino3dSpan11);
clear michelogram
clear delayedSinogram

% Write to a file in interfile format:
outputSinogramName = [outputPath 'delaySinogramSpan11'];
interfileWriteSino(single(delaySinogramSpan11), outputSinogramName, structSizeSino3dSpan11);
% In int 16 also:
outputSinogramName = [outputPath 'delaySinogramSpan11_int16'];
interfileWriteSino(int16(delaySinogramSpan11), outputSinogramName, structSizeSino3dSpan11);
clear delaySinogramSpan11 % Not used in this example.
%% CREATE INITIAL ESTIMATE FOR RECONSTRUCTION
% Create image from the same size than used by siemens:
% Size of the pixels:
sizePixel_mm = [4.1725 4.1725 2.03125];
% The size in pixels:
sizeImage_pixels = [143 143 127]; % For cover the full Fov: 596/4.1725=142.84
% Size of the image to cover the full fov:
sizeImage_mm = sizePixel_mm .* sizeImage_pixels;
% Inititial estimate:
initialEstimate = ones(sizeImage_pixels, 'single');
filenameInitialEstimate = [outputPath '/initialEstimate3d'];
interfilewrite(initialEstimate, filenameInitialEstimate, sizePixel_mm);

% Another image of high resolution:
% Size of the pixels:
sizePixelHighRes_mm = [2.08626 2.08626 2.03125];
sizeImageHighRes_pixels = [285 285 127]; 
% The size in pixels:
%sizeImageHighRes_pixels = [sizeImage_pixels(1)*factor sizeImage_pixels(2)*factor sizeImage_pixels(3)];
% Size of the image to cover the full fov:
sizeImage_mm = sizePixelHighRes_mm .* sizeImageHighRes_pixels;
% Inititial estimate:
initialEstimateHighRes = ones(sizeImageHighRes_pixels, 'single');
filenameInitialEstimateHighRes = [outputPath '/initialEstimate3dHighRes'];
interfilewrite(initialEstimateHighRes, filenameInitialEstimateHighRes, sizePixelHighRes_mm);
%% NORMALIZATION FACTORS
% ncf:
[overall_ncf_3d, scanner_time_invariant_ncf_3d, scanner_time_variant_ncf_3d, used_xtal_efficiencies, used_deadtimefactors, used_axial_factors] = ...
   create_norm_files_mmr('/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/NEMA_IQ_20_02_2014/norm/Norm_20150210112413.n', [], [], [], [], 11);
% invert for nf:
overall_nf_3d = overall_ncf_3d;
overall_nf_3d(overall_ncf_3d ~= 0) = 1./overall_nf_3d(overall_ncf_3d ~= 0);
%% ATTENUATION CORRECTION - PICK A OR B AND COMMENT THE NOT USED 
%% COMPUTE THE ACFS (OPTION A)
% Read the phantom and then generate the ACFs with apirl. There are two
% attenuation maps, the one of the hardware and the one of the patient or
% human.
imageSizeAtten_pixels = [344 344 127];
imageSizeAtten_mm = [2.08626 2.08626 2.03125];
filenameAttenMap_human = '/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/NEMA_IQ_20_02_2014/umap/AttenMapCtManuallyRegistered.i33';
filenameAttenMap_hardware = '/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/NEMA_IQ_20_02_2014/umap/PET_ACQ_194_20150220154553_umap_hardware_00.v';
% Human:
fid = fopen(filenameAttenMap_human, 'r');
if fid == -1
    ferror(fid);
end
attenMap_human = fread(fid, imageSizeAtten_pixels(1)*imageSizeAtten_pixels(2)*imageSizeAtten_pixels(3), 'single');
attenMap_human = reshape(attenMap_human, imageSizeAtten_pixels);
% Then interchange rows and cols, x and y: 
attenMap_human = permute(attenMap_human, [2 1 3]);
fclose(fid);
% The mumap of the phantom it has problems in the spheres, I force all the
% pixels inside the phantom to the same value:

% Hardware:
fid = fopen(filenameAttenMap_hardware, 'r');
if fid == -1
    ferror(fid);
end
attenMap_hardware = fread(fid, imageSizeAtten_pixels(1)*imageSizeAtten_pixels(2)*imageSizeAtten_pixels(3), 'single');
attenMap_hardware = reshape(attenMap_hardware, imageSizeAtten_pixels);
% Then interchange rows and cols, x and y: 
attenMap_hardware = permute(attenMap_hardware, [2 1 3]);
fclose(fid);

% Compose both images:
attenMap = attenMap_hardware + attenMap_human;

% visualization
figure;
image = getImageFromSlices(attenMap, 12, 1, 0);
imshow(image);
title('Attenuation Map Shifted');

% Create ACFs of a computed phatoms with the linear attenuation
% coefficients:
acfFilename = ['acfsSinogramSpan11'];
filenameSinogram = [outputPath 'sinogramSpan11'];
acfsSinogramSpan11 = createACFsFromImage(attenMap, imageSizeAtten_mm, outputPath, acfFilename, filenameSinogram, structSizeSino3dSpan11, 1);
%% READ THE ACFS (OPTION B)
% Span11 Sinogram:
acfFilename = [outputPath 'acfsSinogramSpan11'];
fid = fopen([acfFilename '.i33'], 'r');
numSinos = sum(structSizeSino3dSpan11.sinogramsPerSegment);
[acfsSinogramSpan11, count] = fread(fid, structSizeSino3dSpan11.numTheta*structSizeSino3dSpan11.numR*numSinos, 'single=>single');
acfsSinogramSpan11 = reshape(acfsSinogramSpan11, [structSizeSino3dSpan11.numR structSizeSino3dSpan11.numTheta numSinos]);
% Close the file:
fclose(fid);

%% GENERATE AND SAVE ATTENUATION AND NORMALIZATION FACTORS AND CORECCTION FACTORS FOR SPAN11 SINOGRAMS FOR APIRL
% Save:
outputSinogramName = [outputPath '/NF_Span11'];
interfileWriteSino(single(overall_nf_3d), outputSinogramName, structSizeSino3dSpan11);

% We also generate the ncf:
outputSinogramName = [outputPath '/NCF_Span11'];
interfileWriteSino(single(overall_ncf_3d), outputSinogramName, structSizeSino3dSpan11);

% Compose with acfs:
atteNormFactorsSpan11 = overall_nf_3d;
atteNormFactorsSpan11(acfsSinogramSpan11 ~= 0) = overall_nf_3d(acfsSinogramSpan11 ~= 0) ./acfsSinogramSpan11(acfsSinogramSpan11 ~= 0);
outputSinogramName = [outputPath '/ANF_Span11'];
interfileWriteSino(single(atteNormFactorsSpan11), outputSinogramName, structSizeSino3dSpan11);
clear atteNormFactorsSpan11;
%clear normFactorsSpan11;

% The same for the correction factors:
atteNormCorrectionFactorsSpan11 = overall_ncf_3d .*acfsSinogramSpan11;
outputSinogramName = [outputPath '/ANCF_Span11'];
interfileWriteSino(single(atteNormCorrectionFactorsSpan11), outputSinogramName, structSizeSino3dSpan11);
clear overall_ncf_3d


% Creat a phantom with ones:
onesWithGaps = overall_nf_3d ~= 0;
outputSinogramName = [outputPath '/GAPS_Span11'];
interfileWriteSino(single(onesWithGaps), outputSinogramName, structSizeSino3dSpan11);
clear onesWithGaps
clear overall_nf_3d
%% RANDOMS FROM SINGLES IN BUCKET
sinoRandomsFromSinglesPerBucket = createRandomsFromSinglesPerBucket([filenameUncompressedMmr '.hdr']);
% Create span 11
michelogram = generateMichelogramFromSinogram3D(sinoRandomsFromSinglesPerBucket, structSizeSino3d);
randomsSinogramSpan11 = reduceMichelogram(michelogram, structSizeSino3dSpan11);
clear michelogram
% Write randoms sinogram:
outputSinogramName = [outputPath '/randomsSpan11'];
interfileWriteSino(single(randomsSinogramSpan11), outputSinogramName, structSizeSino3dSpan11);
% Correct for attenuation and normalization (to use in apirl). Since the
% projector is just geomtric, because its cancel with backprojector, the
% randoms and the scatter needs to be multiplied for the normalization
% correction factors and acfs:
randomsSinogramSpan11 = randomsSinogramSpan11 .* atteNormCorrectionFactorsSpan11;
outputSinogramName = [outputPath 'randomsSpan11_ancf'];
interfileWriteSino(single(randomsSinogramSpan11), outputSinogramName, structSizeSino3dSpan11);
%clear atteNormCorrectionFactorsSpan11;
%% GENERATE OSEM AND MLEM RECONSTRUCTION FILES FOR APIRL
% With 1 line.
% Low Res:
numSubsets = 21;
numIterations = 3;
saveInterval = 1;
saveIntermediate = 0;
outputFilenamePrefix = [outputPath sprintf('Nema_Osem%d_LR', numSubsets)];
filenameOsemConfig_LR = [outputPath sprintf('/Osem3dSubset%d_LR.par', numSubsets)];
CreateOsemConfigFileForMmr(filenameOsemConfig_LR, [outputPath 'sinogramSpan11.h33'], [filenameInitialEstimate '.h33'], outputFilenamePrefix, numIterations, [], ...
    numSubsets, saveInterval, saveIntermediate, [], [], [], [outputPath '/ANF_Span11']);

% High Res:
outputFilenamePrefix = [outputPath sprintf('Nema_Osem%d_HR', numSubsets)];
filenameOsemConfig_HR = [outputPath sprintf('/Osem3dSubset%d_HR.par', numSubsets)];
CreateOsemConfigFileForMmr(filenameOsemConfig_HR, [outputPath 'sinogramSpan11.h33'], [filenameInitialEstimateHighRes '.h33'], outputFilenamePrefix, numIterations, [],...
    numSubsets, saveInterval, saveIntermediate, [], [], [], [outputPath '/ANF_Span11.h33']);

% %% RECONSTRUCTION OF HIGH RES IMAGE
% % Execute APIRL:
% status = system(['OSEM ' filenameOsemConfig_HR]) 
% %% GENERATE OSEM AND MLEM RECONSTRUCTION FILES FOR APIRL
% % With 10 lines.
% % Low Res:
% numSubsets = 21;
% numIterations = 3;
% saveInterval = 1;
% saveIntermediate = 0;
% outputFilenamePrefix = [outputPath sprintf('Nema_Osem%d_LR_10lines', numSubsets)];
% filenameOsemConfig_LR = [outputPath sprintf('/Osem3dSubset%d_LR_10lines.par', numSubsets)];
% CreateOsemConfigFileForMmr(filenameOsemConfig_LR, [outputPath 'sinogramSpan11.h33'], [filenameInitialEstimate '.h33'], outputFilenamePrefix, numIterations, [], ...
%     numSubsets, saveInterval, saveIntermediate, [], [], [], [outputPath '/ANF_Span11'], 'siddon number of samples on the detector', 10);
% 
% % High Res:
% outputFilenamePrefix = [outputPath sprintf('Nema_Osem%d_HR_10lines', numSubsets)];
% filenameOsemConfig_HR = [outputPath sprintf('/Osem3dSubset%d_HR_10lines.par', numSubsets)];
% CreateOsemConfigFileForMmr(filenameOsemConfig_HR, [outputPath 'sinogramSpan11.h33'], [filenameInitialEstimateHighRes '.h33'], outputFilenamePrefix, numIterations, [],...
%     numSubsets, saveInterval, saveIntermediate, [], [], [], [outputPath '/ANF_Span11.h33'], 'siddon number of samples on the detector', 10);

%% RECONSTRUCTION OF HIGH RES IMAGE
% Execute APIRL:
status = system(['OSEM ' filenameOsemConfig_HR]) 
% %% GENERATE OSEM AND MLEM RECONSTRUCTION FILES FOR APIRL WITH RANDOMS CORRECTION
% % Low Res:
% numSubsets = 21;
% numIterations = 3;
% saveInterval = 1;
% saveIntermediate = 0;
% outputFilenamePrefix = [outputPath sprintf('Nema_Osem%d_LR_withRandoms', numSubsets)];
% filenameOsemConfig_LR = [outputPath sprintf('/Osem3dSubset%d_LR_withRandoms.par', numSubsets)];
% CreateOsemConfigFileForMmr(filenameOsemConfig_LR, [outputPath 'sinogramSpan11.h33'], [filenameInitialEstimate '.h33'], outputFilenamePrefix, numIterations, [], ...
%     numSubsets, saveInterval, saveIntermediate, [], [], [outputPath 'randomsSpan11_ancf.h33'], [outputPath '/ANF_Span11']);
% 
% % High Res:
% outputFilenamePrefix = [outputPath sprintf('Nema_Osem%d_HR_withRandoms', numSubsets)];
% filenameOsemConfig_HR = [outputPath sprintf('/Osem3dSubset%d_HR_withRandoms.par', numSubsets)];
% CreateOsemConfigFileForMmr(filenameOsemConfig_HR, [outputPath 'sinogramSpan11.h33'], [filenameInitialEstimateHighRes '.h33'], outputFilenamePrefix, numIterations, [],...
%     numSubsets, saveInterval, saveIntermediate, [], [], [outputPath 'randomsSpan11_ancf.h33'], [outputPath '/ANF_Span11.h33']);
% 
% %% RECONSTRUCTION OF HIGH RES IMAGE
% % Execute APIRL:
% status = system(['OSEM ' filenameOsemConfig_HR]) 
%% GENERATE OSEM AND MLEM RECONSTRUCTION FILES FOR APIRL WITH RANDOMS CORRECTION
numIterations = 60;
saveInterval = 1;
outputFilenamePrefix = [outputPath sprintf('Nema_Mlem%d_HR', numSubsets)];
filenameMlemConfig_HR = [outputPath sprintf('/Mlem3dSubset%d_HR.par', numSubsets)];
CreateMlemConfigFileForMmr(filenameMlemConfig_HR, [outputPath 'sinogramSpan11.h33'], [filenameInitialEstimateHighRes '.h33'], outputFilenamePrefix, numIterations, [],...
    saveInterval, saveIntermediate, [], [], [], [outputPath 'ANF_Span11.h33']);

outputFilenamePrefix = [outputPath sprintf('Nema_Mlem%d_HR_10lines', numSubsets)];
filenameMlemConfig_HR_10lines = [outputPath sprintf('/Mlem3dSubset%d_HR_10lines.par', numSubsets)];
CreateMlemConfigFileForMmr(filenameMlemConfig_HR_10lines, [outputPath 'sinogramSpan11.h33'], [filenameInitialEstimateHighRes '.h33'], outputFilenamePrefix, numIterations, [],...
    saveInterval, saveIntermediate, [], [], [], [outputPath 'ANF_Span11.h33'], 'siddon number of samples on the detector', 10);
%% RECONSTRUCTION OF MLEM HIGH RES IMAGE
% Execute APIRL:
status = system(['MLEM ' filenameMlemConfig_HR]) 
% %% RECONSTRUCTION OF MLEM HIGH RES IMAGE
% % Execute APIRL:
% status = system(['MLEM ' filenameMlemConfig_HR_10lines]) 
%% READ RESULTS
% Read interfile reconstructed image:
reconVolume = interfileRead([outputFilenamePrefix '_final.h33']);
% Apply a gaussian 3d filter:
H = fspecial('gaussian',[9 9],1.5); % Two pixel of std dev (aprox 4mm).
filteredVolume = imfilter(reconVolume, H);
% Show slices:
imageToShow = getImageFromSlices(filteredVolume,14);
figure;
imshow(imageToShow);
title(sprintf('Reconstructed Slices for %fx%fx%f mm³ pixel size', sizePixelHighRes_mm(1), sizePixelHighRes_mm(2), sizePixelHighRes_mm(3)));

% Center Slice:
figure;
imshow(filteredVolume(:,:,80)./max(max(filteredVolume(:,:,80))));
colormap(hot);