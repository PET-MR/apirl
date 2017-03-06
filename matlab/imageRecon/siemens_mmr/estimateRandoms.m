%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 09/03/2015
%  *********************************************************************
%  Estimates randoms using the delayed and also estimating the singles.

clear all 
close all

apirlPath = [fileparts(mfilename('fullpath')) filesep '..' filesep '..' filesep '..'];
outputPath = '/home/mab15/workspace/KCL/Biograph_mMr/Randoms/';
%% CUDA PATH
cudaPath = '/usr/local/cuda/';
setenv('PATH', [getenv('PATH') pathsep cudaPath filesep 'bin']);
setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') pathsep cudaPath filesep 'lib64']);
%% STIR PATH
stirPath = '/usr/local/stir3.0/';
stirMatlabPath = '/home/mab15/workspace/KCL/apirl-kcl/trunk/stir/';
scriptsPath = [stirMatlabPath 'scripts/'];
%% SET ENVIRONMENT AND MATLAB PATHS
addpath(genpath([apirlPath filesep 'matlab']));
addpath(genpath(stirMatlabPath));
setenv('PATH', [getenv('PATH') pathsep apirlPath filesep 'build' filesep 'bin' pathsep stirPath filesep 'bin/']);
setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') pathsep apirlPath filesep 'build' filesep 'bin' pathsep stirPath filesep 'lib/' ]);
%% READING THE SINOGRAMS
disp('Read input sinogram...');
% Read the sinograms:
%sinogramFilename = '/home/mab15/workspace/KCL/Biograph_mMr/mmr/5hr_ge68/cylinder_5hours.s.hdr';
sinogramFilename = '/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/BRAIN_PETMR/SINOGRAMS/PET_ACQ_68_20150610155347-0uncomp.s.hdr';
[sinogram, delayedSinogram, structSizeSino3d] = interfileReadSino(sinogramFilename);

%% NORMALIZATION
cbn_filename = '/home/mab15/workspace/KCL/Biograph_mMr/mmr/Norm_20141008101010.n';
[overall_ncf_3d, scanner_time_invariant_ncf_3d, scanner_time_variant_ncf_3d, acquisition_dependant_ncf_3d, used_xtal_efficiencies, used_deadtimefactors, used_axial_factors] = ...
   create_norm_files_mmr(cbn_filename, [], [], [], [], 1);
scanner_time_invariant_ncf_direct = scanner_time_invariant_ncf_3d(1:structSizeSino3d.numZ);

%% SPAN 11
[sinogramSpan11, structSizeSino3dSpan11] = convertSinogramToSpan(sinogram, structSizeSino3d, 1);
[delaySinogramSpan11, structSizeSino3dSpan11] = convertSinogramToSpan(delayedSinogram, structSizeSino3d, 1);

%% DELAYED SINOGRAMS
[ res ] = show_sinos( delaySinogramSpan11, 3, 'delayed span 11', 1 );

%% RANDOM SINOGRAMS SPAN 11 FROM SINGLES IN BUCKET
sinoRandomsFromSinglesPerBucket = createRandomsFromSinglesPerBucket(sinogramFilename);
[componentFactors, componentLabels]  = readmMrComponentBasedNormalization(cbn_filename, 0);
crystalInterfFactor = single(componentFactors{2});
crystalInterfFactor = repmat(crystalInterfFactor', 1, structSizeSino3d.numTheta/size(crystalInterfFactor,1));
% c) Axial factors:
axialFactors = zeros(sum(structSizeSino3d.sinogramsPerSegment),1);
% Generate axial factors from a block profile saved before:
gainPerPixelInBlock = load('axialGainPerPixelInBlock.mat');
numberOfAxialCrystalsPerBlock = 8;
% Check if the sensitivity profle is available:
if exist('sensitivityPerSegment.mat') ~= 0
    sensitivityPerSegment = load('sensitivityPerSegment.mat');
    sensitivityPerSegment = sensitivityPerSegment.sensitivityPerSegment;
else
    sensitivityPerSegment = ones(structSizeSino3d_span1.numSegments,1);
end
if(isstruct(gainPerPixelInBlock))
    gainPerPixelInBlock = gainPerPixelInBlock.gainPerPixelInBlock;
end
indiceSino = 1; % indice del sinogram 3D.
for segment = 1 : structSizeSino3d.numSegments
    % Por cada segmento, voy generando los sinogramas correspondientes y
    % contándolos, debería coincidir con los sinogramas para ese segmento: 
    numSinosThisSegment = 0;
    % Recorro todos los z1 para ir rellenando
    for z1 = 1 : (structSizeSino3d.numZ*2)
        numSinosZ1inSegment = 0;   % Cantidad de sinogramas para z1 en este segmento
        % Recorro completamente z2 desde y me quedo con los que están entre
        % minRingDiff y maxRingDiff. Se podría hacer sin recorrer todo el
        % sinograma pero se complica un poco.
        z1_aux = z1;    % z1_aux la uso para recorrer.
        for z2 = 1 : structSizeSino3d.numZ
            % Ahora voy avanzando en los sinogramas correspondientes,
            % disminuyendo z1 y aumentnado z2 hasta que la diferencia entre
            % anillos llegue a maxRingDiff.
            if ((z1_aux-z2)<=structSizeSino3d.maxRingDiff(segment))&&((z1_aux-z2)>=structSizeSino3d.minRingDiff(segment))
                % Me asguro que esté dentro del tamaño del michelograma:
                if(z1_aux>0)&&(z2>0)&&(z1_aux<=structSizeSino3d.numZ)&&(z2<=structSizeSino3d.numZ)
                    numSinosZ1inSegment = numSinosZ1inSegment + 1;
                    % Get the index of detector inside a block:
                    pixelInBlock1 = rem(z1_aux-1, numberOfAxialCrystalsPerBlock);
                    pixelInBlock2 = rem(z2-1, numberOfAxialCrystalsPerBlock);
                    sinoRandomsFromSinglesPerBucket(:,:,indiceSino) = sinoRandomsFromSinglesPerBucket(:,:,indiceSino) .* crystalInterfFactor;
                    sinoRandomsFromSinglesPerBucket(:,:,indiceSino) = sinoRandomsFromSinglesPerBucket(:,:,indiceSino) .* (gainPerPixelInBlock(pixelInBlock1+1) * gainPerPixelInBlock(pixelInBlock2+1));
                end
            end
            % Pase esta combinación de (z1,z2), paso a la próxima:
            z1_aux = z1_aux - 1;
        end
        if(numSinosZ1inSegment>0)
            % I average the efficencies dividing by the number of axial
            % combinations used for this sino:
            %acquisition_dependant_ncf_3d(:,:,indiceSino) = acquisition_dependant_ncf_3d(:,:,indiceSino) / numSinosZ1inSegment;
            numSinosThisSegment = numSinosThisSegment + 1;
            indiceSino = indiceSino + 1;
        end
    end    
end
%% OTHER SPAN
[randomsSinogramSpan11, structSizeSino3dSpan11] = convertSinogramToSpan(sinoRandomsFromSinglesPerBucket, structSizeSino3d, 11);

% Apply normalization:
[overall_ncf_3d, scanner_time_invariant_ncf_3d, scanner_time_variant_ncf_3d, acquisition_dependant_ncf_3d, used_xtal_efficiencies, used_deadtimefactors, used_axial_factors] = ...
   create_norm_files_mmr(cbn_filename, [], [], [], [], 11);


normalizedRandomsSinogramSpan11 = randomsSinogramSpan11 .* scanner_time_variant_ncf_3d;

%% CREATE RANDOMS ESTIMATE WITH STIR
% The delayed sinogram must be span 1.
[randomsStir, structSizeSino] = estimateRandomsWithStir(delayedSinogram, structSizeSino3d, overall_ncf_3d, structSizeSino3dSpan11, outputPath);
%% CREATE RANDOMS WITH MY FUNCTION
numIterations = 3;
[randomsFromDelayeds, singlesOut] = estimateRandomsFromDelayeds(delayedSinogram, structSizeSino3d, numIterations, 11);
%% WRITE OUTPUT
interfileWriteSino(single(randomsStir), [outputPath 'stirRandoms'], structSizeSino);
interfileWriteSino(single(randomsFromDelayeds), [outputPath 'apirlRandoms'], structSizeSino);
%%
sinogramPlane = 307;
figure;
plot([delaySinogramSpan11(:, 128,sinogramPlane) randomsSinogramSpan11(:, 128,sinogramPlane) normalizedRandomsSinogramSpan11(:, 128,sinogramPlane) ...
    randomsStir(:, 128,sinogramPlane) randomsFromDelayeds(:, 128,sinogramPlane)]);
legend('Delayed', 'Randoms from Singles', 'Randoms from Singles Normalized', 'Randoms from Stir', 'Randoms from Delayeds'); 
%%
figure;
plot([delaySinogramSpan11(:, 128,sinogramPlane) randomsSinogramSpan11(:, 128,sinogramPlane) normalizedRandomsSinogramSpan11(:, 128,sinogramPlane) ...
    randomsStir(:, 128,sinogramPlane) randomsFromDelayeds(:, 128,sinogramPlane)./mean(randomsFromDelayeds(:)).*mean(randomsStir(:))]);
legend('Delayed', 'Randoms from Singles', 'Randoms from Singles Normalized', 'Randoms from Stir', 'Randoms from Delayeds'); 
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
