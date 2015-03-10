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
% Read the sinograms:
sinogramsPath = '/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/NEMA_IQ_20_02_2014/';
filenameUncompressedMmr = [sinogramsPath 'PET_ACQ_194_20150220154553-0uncomp.s'];
outFilenameIntfSinograms = [sinogramsPath 'NemaIq20_02_2014_ApirlIntf.s'];
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

%% DELAYED SINOGRAMS FROM SINGLES IN BUCKET
sinoRandomsFromSinglesPerBucket = createRandomsFromSinglesPerBucket([filenameUncompressedMmr '.hdr']);
michelogram = generateMichelogramFromSinogram3D(sinoRandomsFromSinglesPerBucket, structSizeSino3d);
% Plot direct delayed sinograms:
imagesToShow = getImageFromSlices(sinoRandomsFromSinglesPerBucket(:,:,1:structInterfile.NumberOfRings), 10);
figure;
imshow(imagesToShow);
title('Estimated Randoms From Singles per Bucket for Direct Sinograms (Span 1)');
set(gcf, 'Position', [0 0 1600 1200]);
% Create span 11
delaySinogramSpan11 = reduceMichelogram(michelogram, structSizeSino3dSpan11);
clear michelogram