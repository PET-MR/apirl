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
disp('Read input sinogram...');
% Read the sinograms:
sinogramFilename = '/home/mab15/workspace/KCL/Biograph_mMr/mmr/5hr_ge68/cylinder_5hours.s.hdr';
[sinogram, delayedSinogram, structSizeSino3d] = interfileReadSino(sinogramFilename);

%% NORMALIZATION
cbn_filename = '/home/mab15/workspace/KCL/Biograph_mMr/mmr/Norm_20141008101010.n';
[overall_ncf_3d, scanner_time_invariant_ncf_3d, scanner_time_variant_ncf_3d, acquisition_dependant_ncf_3d, used_xtal_efficiencies, used_deadtimefactors, used_axial_factors] = ...
   create_norm_files_mmr(cbn_filename, [], [], [], [], 1);
scanner_time_invariant_ncf_direct = scanner_time_invariant_ncf_3d(1:structSizeSino3d.numZ);

%% SPAN 11
[sinogramSpan11, structSizeSino3dSpan11] = convertSinogramToSpan(sinogram, structSizeSino3d, 11);
[delaySinogramSpan11, structSizeSino3dSpan11] = convertSinogramToSpan(delayedSinogram, structSizeSino3d, 11);

%% DELAYED SINOGRAMS
[ res ] = show_sinos( delaySinogramSpan11, 3, 'delayed span 11', 1 );

%% RANDOM SINOGRAMS SPAN 11 FROM SINGLES IN BUCKET
sinoRandomsFromSinglesPerBucket = createRandomsFromSinglesPerBucket(sinogramFilename);
[randomsSinogramSpan11, structSizeSino3dSpan11] = convertSinogramToSpan(sinoRandomsFromSinglesPerBucket, structSizeSino3d, 11);

% Apply normalization:
[overall_ncf_3d, scanner_time_invariant_ncf_3d, scanner_time_variant_ncf_3d, acquisition_dependant_ncf_3d, used_xtal_efficiencies, used_deadtimefactors, used_axial_factors] = ...
   create_norm_files_mmr(cbn_filename, [], [], [], [], 11);
normalizedRandomsSinogramSpan11 = randomsSinogramSpan11 .* scanner_time_variant_ncf_3d;

%% CREATE RANDOMS ESTIMATE WITH STIR
% The delayed sinogram must be span 1.
[randomsStir, structSizeSino] = estimateRandomsWithStir(delayedSinogram, structSizeSino3d, overall_ncf_3d, structSizeSino3dSpan11, outputPath);

%%
figure;
aux = mean(delaySinogramSpan11,3);
aux2 = mean(randomsSinogramSpan11,3);
aux3 = mean(normalizedRandomsSinogramSpan11,3);
aux4 = mean(randomsStir,3);
plot([aux(:, 128) aux2(:, 128) aux3(:, 128) aux4(:, 128)]);

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
