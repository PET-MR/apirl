
%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 10/02/2015
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
outputPath = '/home/mab15/workspace/KCL/Biograph_mMr/mmr/5hr_ge68/span11/';
mkdir(outputPath);
%% TEST NORM FUNCTION
% ncf:
[overall_ncf_3d, scanner_time_invariant_ncf_3d, scanner_time_variant_ncf_3d, used_xtal_efficiencies, used_deadtimefactors, used_axial_factors] = ...
   create_norm_files_mmr('/home/mab15/workspace/KCL/Biograph_mMr/mmr/Norm_20141008101010.n', [], [], [], [], 11);
% invert for nf:
overall_nf_3d = overall_ncf_3d;
overall_nf_3d(overall_ncf_3d ~= 0) = 1./overall_nf_3d(overall_ncf_3d ~= 0);
% Size of mMr Sinogram's
numTheta = 252; numR = 344; numRings = 64; maxAbsRingDiff = 60; rFov_mm = 594/2; zFov_mm = 258; span = 11;
structSizeSino3dSpan11 = getSizeSino3dFromSpan(numR, numTheta, numRings, rFov_mm, zFov_mm, span, maxAbsRingDiff);

outputSinogramName = [outputPath '/NCF_Span11_withFunc'];
interfileWriteSino(single(overall_ncf_3d), outputSinogramName, structSizeSino3dSpan11.sinogramsPerSegment, structSizeSino3dSpan11.minRingDiff, structSizeSino3dSpan11.maxRingDiff);

outputSinogramName = [outputPath '/NF_Span11_withFunc'];
interfileWriteSino(single(overall_nf_3d), outputSinogramName, structSizeSino3dSpan11.sinogramsPerSegment, structSizeSino3dSpan11.minRingDiff, structSizeSino3dSpan11.maxRingDiff);
%% COMPLETE THE FACTORS WITH ATTENUATION
% Span11 Sinogram:
acfFilename = [outputPath 'acfsSinogramSpan11'];
fid = fopen([acfFilename '.i33'], 'r');
numSinos = sum(structSizeSino3dSpan11.sinogramsPerSegment);
[acfsSinogramSpan11, count] = fread(fid, structSizeSino3dSpan11.numTheta*structSizeSino3dSpan11.numR*numSinos, 'single=>single');
acfsSinogramSpan11 = reshape(acfsSinogramSpan11, [structSizeSino3dSpan11.numR structSizeSino3dSpan11.numTheta numSinos]);
% Close the file:
fclose(fid);
%% ANCF
ancf = overall_ncf_3d .* acfsSinogramSpan11;
outputSinogramName = [outputPath '/ANCF_Span11_withFunc'];
interfileWriteSino(single(ancf), outputSinogramName, structSizeSino3dSpan11.sinogramsPerSegment, structSizeSino3dSpan11.minRingDiff, structSizeSino3dSpan11.maxRingDiff);
% invert for anf:
anf = ancf;
anf(ancf ~= 0) = 1./ancf(ancf ~= 0);
outputSinogramName = [outputPath '/ANF_Span11_withFunc'];
interfileWriteSino(single(anf), outputSinogramName, structSizeSino3dSpan11.sinogramsPerSegment, structSizeSino3dSpan11.minRingDiff, structSizeSino3dSpan11.maxRingDiff);