%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 11/05/2015
%  *********************************************************************
%  This function reconstructs an image using the self normalization
%  reconstruction algorithm. Uses a time invariant normalization from a .n
%  file and estimates the crystal efficiencies. It optionally receives the
%  filename of the .mat system matrixes for computing the crystal
%  efficiencies:
%  % Examples:
%   [volume, xtal_efficiencies] = OsemMmrSpan1(sinogramFilename, normFilename, attMapBaseFilename, outputPath, [2.08625 2.08625 2.03125], 21, 3, 3)
%   [volume, xtal_efficiencies] = OsemMmrSpan1(sinogramFilename, normFilename, attMapBaseFilename, outputPath, [2.08625 2.08625 2.03125], 21, 3, 3, 'detector1SystemMatrixSpan1.mat','detector2SystemMatrixSpan1.mat')


function [volume, xtal_efficiencies] = SelfNormRecon(sinogramFilename, normFilename, attMapBaseFilename, outputPath, pixelSize_mm, numOsemSubsets, numOsemIterations, numXtalIterations, filenameSystemMatrix1, filenameSystemMatrix2)

mkdir(outputPath);
% Check what OS I am running on:
if(strcmp(computer(), 'GLNXA64'))
    os = 'linux';
    pathBar = '/';
elseif(strcmp(computer(), 'PCWIN') || strcmp(computer(), 'PCWIN64'))
    os = 'windows';
    pathBar = '\';
else
    disp('OS not compatible');
    return;
end

%% READING THE SINOGRAMS
disp('Converting input sinogram to APIRL format...');
% Read the sinograms. Check if its siemens or apirl:
[pathstr,name,ext] = fileparts(sinogramFilename);
outFilenameIntfSinograms = [outputPath pathBar 'inputSinogramSpan1'];
[info, structSizeSino3d] = getInfoFromSiemensIntf([sinogramFilename '.h33']);
if isempty(structSizeSino3d)
    % apirl sinogram. Nothing to do. Just read the raw data:% Size of mMr Sinogram's
    numTheta = 252; numR = 344; numRings = 64; maxAbsRingDiff = 60; rFov_mm = 594/2; zFov_mm = 258; span = 1;
    structSizeSino3d = getSizeSino3dFromSpan(numR, numTheta, numRings, rFov_mm, zFov_mm, span, maxAbsRingDiff);
    fid = fopen([sinogramFilename '.i33'], 'r');
    numSinos = sum(structSizeSino3d.sinogramsPerSegment);
    [sinogram, count] = fread(fid, structSizeSino3d.numTheta*structSizeSino3d.numR*numSinos, 'single=>single');
    sinogram = reshape(sinogram, [structSizeSino3d.numR structSizeSino3d.numTheta numSinos]);
    % Close the file:
    fclose(fid);
    
    interfileWriteSino(sinogram, outFilenameIntfSinograms, structSizeSino3d)
else
    [sinogram, delaySinogram, structSizeSino3d] = getIntfSinogramsFromUncompressedMmr(sinogramFilename, outFilenameIntfSinograms);
    
end
% Rewrite the sinogram filename to be used in the next operations:
sinogramFilename = [outFilenameIntfSinograms];

% At the moment, only span 1.
% % Generate the span sinogram if necessary:
% michelogram = generateMichelogramFromSinogram3D(sinogramSpan1, structSizeSino3d);
% structSizeSino3d = getSizeSino3dFromSpan(structSizeSino3d.numR, structSizeSino3d.numTheta, structSizeSino3d.numZ, ...
%     structSizeSino3d.rFov_mm, structSizeSino3d.zFov_mm, 11, structSizeSino3d.maxAbsRingDiff);
% sinogramSpan11 = reduceMichelogram(michelogram, structSizeSino3dSpan11);
% clear michelogram
% clear sinogram
% % Write to a file in interfile format:
% outputSinogramName = [outputPath 'sinogramSpan11'];
% interfileWriteSino(single(sinogramSpan11), outputSinogramName, structSizeSino3dSpan11);
%% PROCESS INPUT PARAMETERS
% Image size:
imageSize_mm = [600 600 258];
imageSize_pixels = ceil(imageSize_mm./pixelSize_mm);

if nargin == 8
    [detector1SystemMatrix, detector2SystemMatrix] = createDetectorSystemMatrix3d(span, 1);
elseif nargin == 10
    detector1SystemMatrix = load(filenameSystemMatrix1);
    detector2SystemMatrix = load(filenameSystemMatrix2);
    if isstruct(detector1SystemMatrix)
        detector1SystemMatrix = detector1SystemMatrix.detector1SystemMatrix;
        detector2SystemMatrix = detector2SystemMatrix.detector2SystemMatrix;
    end
else
    error('Wrong number of parameters: [volume, xtal_efficiencies] = OsemMmrSpan1(sinogramFilename, normFilename, attMapBaseFilename, outputPath, [2.08625 2.08625 2.03125], 21, 3, 3, ''detector1SystemMatrixSpan1.mat'',''detector2SystemMatrixSpan1.mat'')');
end    


%% CREATE INITIAL ESTIMATE FOR RECONSTRUCTION
disp('Creating inital image...');
% Inititial estimate:
initialEstimate = ones(imageSize_pixels, 'single');
filenameInitialEstimate = [outputPath pathBar 'initialEstimate'];
interfilewrite(initialEstimate, filenameInitialEstimate, pixelSize_mm);
%% NORMALIZATION FACTORS
disp('Computing the normalization correction factors...');
% ncf:
[overall_ncf_3d, scanner_time_invariant_ncf_3d, scanner_time_variant_ncf_3d, used_xtal_efficiencies, used_deadtimefactors, used_axial_factors] = ...
   create_norm_files_mmr(normFilename, [], [], [], [], structSizeSino3d.span);
% invert for nf:
overall_nf_3d = overall_ncf_3d;
overall_nf_3d(overall_ncf_3d ~= 0) = 1./overall_nf_3d(overall_ncf_3d ~= 0);
%% ATTENUATION MAP
if ~strcmp(attMapBaseFilename, '')
    disp('Computing the attenuation correction factors...');
    % Read the attenuation map and compute the acfs.
    headerInfo = getInfoFromSiemensIntf([attMapBaseFilename '_umap_human_00.v.hdr']);
    headerInfo = getInfoFromSiemensIntf([attMapBaseFilename '_umap_hardware_00.v.hdr']);
    imageSizeAtten_pixels = [headerInfo.MatrixSize1 headerInfo.MatrixSize2 headerInfo.MatrixSize3];
    imageSizeAtten_mm = [headerInfo.ScaleFactorMmPixel1 headerInfo.ScaleFactorMmPixel2 headerInfo.ScaleFactorMmPixel3];
    filenameAttenMap_human = [attMapBaseFilename '_umap_human_00.v'];
    filenameAttenMap_hardware = [attMapBaseFilename '_umap_hardware_00.v'];
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

    % Create ACFs of a computed phatoms with the linear attenuation
    % coefficients:
    acfFilename = ['acfsSinogram'];
    acfsSinogram = createACFsFromImage(attenMap, imageSizeAtten_mm, outputPath, acfFilename, sinogramFilename, structSizeSino3d, 0);

    % After the projection read the acfs:
    acfFilename = [outputPath acfFilename];
    fid = fopen([acfFilename '.i33'], 'r');
    numSinos = sum(structSizeSino3d.sinogramsPerSegment);
    [acfsSinogram, count] = fread(fid, structSizeSino3d.numTheta*structSizeSino3d.numR*numSinos, 'single=>single');
    acfsSinogram = reshape(acfsSinogramSpan1, [structSizeSino3d.numR structSizeSino3d.numTheta numSinos]);
    % Close the file:
    fclose(fid);
else
    acfFilename = '';
    acfsSinogram = ones(size(sinogram), 'single');
end
%% GENERATE AND SAVE ATTENUATION AND NORMALIZATION FACTORS AND CORECCTION FACTORS FOR SPAN11 SINOGRAMS FOR APIRL
disp('Generating the ANF sinogram...');
% Save:
outputSinogramName = [outputPath '/NF'];
interfileWriteSino(single(overall_nf_3d), outputSinogramName, structSizeSino3d);

% Compose with acfs:
atteNormFactors = overall_nf_3d;
atteNormFactors(acfsSinogram ~= 0) = overall_nf_3d(acfsSinogram ~= 0) ./acfsSinogram(acfsSinogram ~= 0);
anfFilename = [outputPath '/ANF'];
interfileWriteSino(single(atteNormFactors), anfFilename, structSizeSino3d);
clear atteNormFactors;
%clear normFactorsSpan11;
%% SELF NORMALIZING ALGORITHM

% Initialize crystal efficiencies:
numDetectorsPerRing = 504;
numRings = 64;
crystalEfficiencies = ones(numDetectorsPerRing, numRings); % Initialize the crystal efficiencies with ones.

disp('########## STARTING ITERATIVE ALGORITHM ##########');
% Iterative process:
for iteration = 1 : numXtalIterations
    disp(sprintf('Iteration No %d', iteration));
    disp(sprintf('Compute NCF in iteration %d', iteration));
    % 1) Compute normalization factors with the crystal efficiencies:
    [overall_ncf_3d, scanner_time_invariant_ncf_3d, scanner_time_variant_ncf_3d, acquisition_dependant_ncf_3d, crystalEfficiencies, used_deadtimefactors, used_axial_factors] = ...
        create_norm_files_mmr(normFilename, [], crystalEfficiencies, [], [], structSizeSino3d.span);  % Recompute normalization factors with the crystal efficiencies.
    save([outputPath sprintf('crystalEfficiencies_iter%d', iteration)], 'crystalEfficiencies'); % Save crystal eff.
    crystalEffCorrFactorsFilename = [outputPath sprintf('CrystalEffCorrFactors_iter%d', iteration)]; % Save crystal correction factors.
    % invert for nf:
    overall_nf_3d = overall_ncf_3d;
    overall_nf_3d(overall_ncf_3d ~= 0) = 1./overall_nf_3d(overall_ncf_3d ~= 0);
    % now generate anf:
    atteNormFactors = overall_nf_3d;
    atteNormFactors(acfsSinogram ~= 0) = overall_nf_3d(acfsSinogram ~= 0) ./acfsSinogram(acfsSinogram ~= 0);
    anfFilename = [outputPath sprintf('ANF_iter%d', iteration)];
    interfileWriteSino(single(atteNormFactors), anfFilename, structSizeSino3d);

    % 2) Reconstruct with 1 iteration:
    disp('Osem reconstruction...');
    saveInterval = 1;
    saveIntermediate = 0;
    outputFilenamePrefix = [outputPath sprintf('reconImage_iter%d',iteration)];
    filenameOsemConfig = [outputPath sprintf('osem_iter%d.par', iteration)];
    CreateOsemConfigFileForMmr(filenameOsemConfig, [sinogramFilename '.h33'], [filenameInitialEstimate '.h33'], outputFilenamePrefix, numOsemIterations, [],...
        numOsemSubsets, saveInterval, saveIntermediate, [], [], [], [anfFilename '.h33']);
    % Execute APIRL:
    status = system(['OSEM ' filenameOsemConfig]) 
    % Show image:
    reconImageFilename = [outputFilenamePrefix '_final.h33'];
    reconVolume = interfileRead(reconImageFilename);
    % Show slices:
    figure;
    imshow(reconVolume(:,:,81)./max(max(reconVolume(:,:,81))));
    title(sprintf('Reconstructed Image in Iteration %d', iteration));

    % 3) Project the reconstructed image:
    filenameProjectionConfig = [outputPath sprintf('projectReconstructedImage_iter%d.par', iteration)];
    outputSample = [sinogramFilename '.h33'];
    projectionFilename = [outputPath sprintf('projectedReconstructedImage_iter%d', iteration)];
    %CreateProjectConfigFileForMmr(filenameProjectionConfig, [reconFilteredImageFilename '.h33'], outputSample, projectionFilename);
    CreateProjectConfigFileForMmr(filenameProjectionConfig, reconImageFilename, outputSample, projectionFilename);
    status = system(['project ' filenameProjectionConfig])

    % 4) Ratio between sinogram and projected:
    fid = fopen([projectionFilename '.i33'], 'r'); % Read projected sinogram:
    [projectedImage, count] = fread(fid, structSizeSino3d.numTheta*structSizeSino3d.numR*numSinos, 'single=>single');
    projectedImage = reshape(projectedImage, [structSizeSino3d.numR structSizeSino3d.numTheta numSinos]);
    % Close the file:
    fclose(fid);
    % Apply normalization and attenuation without crystal efficiencies:
    projectedImage_nf = projectedImage;
    projectedImage_nf = projectedImage_nf .*atteNormFactors;

    % 5) Estimate crystal efficiencies:    
    % Other method:
    useOfDetectorsInSinogram = detector1SystemMatrix'*double(sinogram(:)) + detector2SystemMatrix'*double(sinogram(:));
    useOfDetectorsInProjection = detector1SystemMatrix'*double(projectedImage_nf(:)) + detector2SystemMatrix'*double(projectedImage_nf(:));
    % Ratio of use of detectors:
    crystalEfficienciesCorrection = useOfDetectorsInSinogram./useOfDetectorsInProjection;
    
    
    save([outputPath sprintf('crystalEfficienciesCorrection_iter%d', iteration)], 'crystalEfficienciesCorrection', '-v7.3' ); % Save crystal eff.
    crystalEfficienciesVector = crystalEfficiencies(:) .* crystalEfficienciesCorrection;
    % 6) Save and compare qith the current crystal efficiencies:
    figure;
    plot(1:numDetectors, simulatedXtalEff(:), 1:numDetectors, crystalEfficienciesVector, 1:numDetectors, crystalEfficienciesCorrection);
    legend('Ground Truth', 'New Crystal Efficiencies', 'Crystal Efficiencies Correction');
%     % The crystal efficiences that were zeros because there were no counts,
%     % I left it with it's original value:
%     crystalEfficienciesVector(crystalEfficienciesVector == 0) = 1;
    % 7) Reshape crystal efficiences for the original format (by ring):
    crystalEfficiencies = reshape(crystalEfficienciesVector, [numDetectorsPerRing numRings]);
end
save([outputPath 'crystalEfficiencies_final'], 'crystalEfficiencies'); % Save crystal eff.