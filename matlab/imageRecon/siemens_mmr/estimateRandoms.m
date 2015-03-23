%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 09/03/2015
%  *********************************************************************
%  Estimates randoms using the delayed and also estimating the singles.

clear all 
close all
%% PATHS FOR EXTERNAL FUNCTIONS AND RESULTS
addpath('/home/mab15/workspace/KCL/Biograph_mMr/mmr');
apirlPath = '/home/mab15/workspace/apirl-code/trunk/';
addpath(genpath([apirlPath '/matlab']));
setenv('PATH', [getenv('PATH') ':/home/mab15/workspace/apirl-code/trunk/build/bin']);
setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') ':/home/mab15/workspace/apirl-code/trunk/build/bin']);
outputPath = '/home/mab15/workspace/KCL/Biograph_mMr/Randoms/';
mkdir(outputPath);
%setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') ':/usr/lib/x86_64-linux-gnu/']);
%% READING THE SINOGRAMS
% Read the 5 hours sinogram:
sinogramsPath = '/home/mab15/workspace/KCL/Biograph_mMr/mmr/5hr_ge68/';
filenameUncompressedMmr = [sinogramsPath 'cylinder_5hours.s'];
outFilenameIntfSinograms = [sinogramsPath 'cylinder_5hoursIntf.s'];
[structInterfile, structSizeSino] = getInfoFromSiemensIntf([filenameUncompressedMmr '.hdr']);
[sinogram, delayedSinogram, structSizeSino3d] = getIntfSinogramsFromUncompressedMmr(filenameUncompressedMmr, outFilenameIntfSinograms);

%% NORMALIZATION
cbn_filename = '/home/mab15/workspace/KCL/Biograph_mMr/mmr/Norm_20141008101010.n';
[overall_ncf_3d, scanner_time_invariant_ncf_3d, scanner_time_variant_ncf_3d, acquisition_dependant_ncf_3d, used_xtal_efficiencies, used_deadtimefactors, used_axial_factors] = ...
   create_norm_files_mmr(cbn_filename, [], [], [], [], 1);
scanner_time_invariant_ncf_direct = scanner_time_invariant_ncf_3d(1:structSizeSino3d.numZ);

%% SPAN 11
% The same for the delayed:
% Create sinogram span 11:
structSizeSino3dSpan11 = getSizeSino3dFromSpan(structSizeSino3d.numR, structSizeSino3d.numTheta, structSizeSino3d.numZ, ...
    structSizeSino3d.rFov_mm, structSizeSino3d.zFov_mm, 11, structSizeSino3d.maxAbsRingDiff);
michelogram = generateMichelogramFromSinogram3D(delayedSinogram, structSizeSino3d);
delaySinogramSpan11 = reduceMichelogram(michelogram, structSizeSino3dSpan11);
clear michelogram
% Normalize to cps:
delaySinogramSpan11 = delaySinogramSpan11; %The randoms estimate is already in counts ./ structInterfile.ImageDurationSec;
outputSinogramName = [outputPath 'delaySpan11'];
interfileWriteSino(single(delaySinogramSpan11), outputSinogramName, structSizeSino3dSpan11);
%% DELAYED SINOGRAMS
% Plot a few delayed sinograms:
imagesToShow = getImageFromSlices(delaySinogramSpan11(:,:,1:structInterfile.NumberOfRings*2-1), 12);
figure;
imshow(imagesToShow);
title('Delayed Sinograms for Direct Sinograms (Span 11)');
set(gcf, 'Position', [0 0 1600 1200]);
% Plot th mean delayed direct sinograms:
figure;
imshow(mean(delaySinogramSpan11(:,:,1:structInterfile.NumberOfRings*2-1),3));
title('Mean Delayed Sinograms for Direct Sinograms (Span 11)');
set(gcf, 'Position', [0 0 1600 1200]);

%% RANDOM SINOGRAMS SPAN 11 FROM SINGLES IN BUCKET
sinoRandomsFromSinglesPerBucket = createRandomsFromSinglesPerBucket([filenameUncompressedMmr '.hdr']);
michelogram = generateMichelogramFromSinogram3D(sinoRandomsFromSinglesPerBucket, structSizeSino3d);
% Plot direct delayed sinograms:
imagesToShow = getImageFromSlices(sinoRandomsFromSinglesPerBucket(:,:,1:structInterfile.NumberOfRings), 10);
figure;
imshow(imagesToShow);
title('Estimated Randoms From Singles per Bucket for Direct Sinograms (Span 1)');
set(gcf, 'Position', [0 0 1600 1200]);
% Create span 11
randomsSinogramSpan11 = reduceMichelogram(michelogram, structSizeSino3dSpan11);
clear michelogram
% Write randoms sinogram:
outputSinogramName = [outputPath '/randomsSpan11'];
interfileWriteSino(single(randomsSinogramSpan11), outputSinogramName, structSizeSino3dSpan11);

% Apply normalization:
[overall_ncf_3d, scanner_time_invariant_ncf_3d, scanner_time_variant_ncf_3d, acquisition_dependant_ncf_3d, used_xtal_efficiencies, used_deadtimefactors, used_axial_factors] = ...
   create_norm_files_mmr(cbn_filename, [], [], [], [], 11);
randomsSinogramSpan11 = randomsSinogramSpan11 .* scanner_time_variant_ncf_3d;

outputSinogramName = [outputPath 'randomsSpan11_ncf'];
interfileWriteSino(single(randomsSinogramSpan11), outputSinogramName, structSizeSino3dSpan11);
outputSinogramName = [outputPath 'used_ncf'];
interfileWriteSino(single(scanner_time_variant_ncf_3d), outputSinogramName, structSizeSino3dSpan11);
%% PLOT PROFILES
figure;
plot([randomsSinogramSpan11(:,180,10) delaySinogramSpan11(:,180,10)]);

randomsPerSlice = sum(sum(randomsSinogramSpan11));
randomsPerSlice = permute(randomsPerSlice, [3 1 2]);
delaysPerSlice = sum(sum(delaySinogramSpan11));
delaysPerSlice = permute(delaysPerSlice, [3 1 2]);
figure;
plot([randomsPerSlice delaysPerSlice]);
%% WITH AXIAL NORMALIZATION FACTORS
[componentFactors, componentLabels]  = readmMrComponentBasedNormalization(cbn_filename, 0);
figure;
title('Estimated Randoms From Singles per Bucket for Span 11 with Axial Correction Factors');
set(gcf, 'Position', [0 0 1600 1200]);
plot([randomsPerSlice delaysPerSlice delaysPerSlice.*componentFactors{4}.*componentFactors{8}], 'LineWidth', 2);
legend('Randoms', 'Delays', 'Delays Axial Factors 1-2');
