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


function [reconVolume, xtal_efficiencies] = SelfNormRecon(siemensSinogramFilename, normFilename, attMapBaseFilename, randomsFilename, scatterFilename, outputPath, pixelSize_mm, numOsemSubsets, numOsemIterations, numXtalIterations, filenameSystemMatrix1, filenameSystemMatrix2, stirMatlabPath)

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

useGpu = 1;
%% READING THE SINOGRAMS
disp('Read input sinogram...');
% Read the sinograms:
[sinogram, delayedSinograms, structSizeSino3dSpan1] = interfileReadSino(siemensSinogramFilename);
% % Convert to span: Only span 1
% [sinograms, structSizeSino3d] = convertSinogramToSpan(sinograms, structSizeSino3dSpan1, span);
span = 1;
structSizeSino3d = structSizeSino3dSpan1;
sinogramFilename = [outputPath pathBar 'sinogram'];
% Write the input sinogram:
interfileWriteSino(single(sinogram), sinogramFilename, structSizeSino3dSpan1);

%% PROCESS INPUT PARAMETERS
% Image size:
imageSize_mm = [600 600 258];
imageSize_pixels = ceil(imageSize_mm./pixelSize_mm);

if isempty(filenameSystemMatrix1) || isempty(filenameSystemMatrix2)
    disp('Computing detector system matrix...');
    [detector1SystemMatrix, detector2SystemMatrix] = createDetectorSystemMatrix3d(span, 1);
    save([outputPath pathBar 'detector1SystemMatrix'],'detector1SystemMatrix', '-v7.3');
    save([outputPath pathBar 'detector2SystemMatrix'],'detector2SystemMatrix', '-v7.3');
else
    disp('Loading detector system matrix...');
    detector1SystemMatrix = load(filenameSystemMatrix1);
    detector2SystemMatrix = load(filenameSystemMatrix2);
    if isstruct(detector1SystemMatrix)
        detector1SystemMatrix = detector1SystemMatrix.detector1SystemMatrix;
        detector2SystemMatrix = detector2SystemMatrix.detector2SystemMatrix;
    end
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
[overall_ncf_3d, scanner_time_invariant_ncf_3d, scanner_time_variant_ncf_3d, acquisition_dependant_ncf_3d, original_xtal_efficiencies, used_deadtimefactors, used_axial_factors] = ...
   create_norm_files_mmr(normFilename, [], [], [], [], structSizeSino3d.span);
% invert for nf:
overall_nf_3d = overall_ncf_3d;
overall_nf_3d(overall_ncf_3d ~= 0) = 1./overall_nf_3d(overall_ncf_3d ~= 0);
%% ATTENUATION MAP
if ~strcmp(attMapBaseFilename, '')
    % Check if its attenuation from siemens or a post processed image:
    if ~strcmp(attMapBaseFilename(end-3:end),'.h33')
        disp('Computing the attenuation correction factors from mMR mu maps...');
        % Read the attenuation map and compute the acfs.
        headerInfo = getInfoFromInterfile([attMapBaseFilename '_umap_human_00.v.hdr']);
        headerInfo = getInfoFromInterfile([attMapBaseFilename '_umap_hardware_00.v.hdr']);
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
    
    else
        disp('Computing the attenuation correction factors from post processed APIRL mu maps...');
        attenMap = interfileRead(attMapBaseFilename);
        infoAtten = interfileinfo(attMapBaseFilename); 
        imageSizeAtten_mm = [infoAtten.ScalingFactorMmPixel1 infoAtten.ScalingFactorMmPixel2 infoAtten.ScalingFactorMmPixel3];
    end
    % Create ACFs of a computed phatoms with the linear attenuation
    % coefficients:
    acfFilename = ['acfsSinogram'];
    acfsSinogram = createACFsFromImage(attenMap, imageSizeAtten_mm, outputPath, acfFilename, sinogramFilename, structSizeSino3d, 0, useGpu);

    % After the projection read the acfs:
    acfFilename = [outputPath acfFilename];
    fid = fopen([acfFilename '.i33'], 'r');
    numSinos = sum(structSizeSino3d.sinogramsPerSegment);
    [acfsSinogram, count] = fread(fid, structSizeSino3d.numTheta*structSizeSino3d.numR*numSinos, 'single=>single');
    acfsSinogram = reshape(acfsSinogram, [structSizeSino3d.numR structSizeSino3d.numTheta numSinos]);
    % Close the file:
    fclose(fid);
else
    acfFilename = '';
end
%% CREATE MULTIPLICATIVE FACTORS
% Compose with acfs:
atteNormFactors = overall_nf_3d;
atteNormFactors(acfsSinogram ~= 0) = overall_nf_3d(acfsSinogram ~= 0) ./acfsSinogram(acfsSinogram ~= 0);
%% SELF NORMALIZING ALGORITHM

% Initialize crystal efficiencies:
numDetectorsPerRing = 504;
numRings = 64;
numDetectors = numDetectorsPerRing*numRings;
crystalEfficiencies = ones(numDetectorsPerRing, numRings); % Initialize the crystal efficiencies with ones.
crystalEfficiencies = 2.* rand(size(original_xtal_efficiencies));
crystalEfficiencies(original_xtal_efficiencies == 0) = 0;
scatter = 1;
randoms = 1;
saveInterval = 1;
useGpu = 1;
removeTempFiles = 1;
disp('########## STARTING ITERATIVE ALGORITHM ##########');
% Iterative process:
for iteration = 1 : numXtalIterations
    outputPathIter = [outputPath sprintf('Iteration_%d/', iteration)];
    if ~isdir(outputPathIter)
        mkdir(outputPathIter)
    end
    disp(sprintf('Iteration No %d', iteration));
    disp(sprintf('Compute NCF in iteration %d', iteration));
    % 1) Compute normalization factors with the crystal efficiencies:
    [overall_ncf_3d, scanner_time_invariant_ncf_3d, scanner_time_variant_ncf_3d, acquisition_dependant_ncf_3d, crystalEfficiencies, used_deadtimefactors, used_axial_factors] = ...
        create_norm_files_mmr(normFilename, [], crystalEfficiencies, [], [], structSizeSino3d.span);  % Recompute normalization factors with the crystal efficiencies.
    save([outputPathIter sprintf('crystalEfficiencies_iter%d', iteration)], 'crystalEfficiencies'); % Save crystal eff.
    crystalEffCorrFactorsFilename = [outputPathIter sprintf('CrystalEffCorrFactors_iter%d', iteration)]; % Save crystal correction factors.

    [volume overall_ncf_3d acfsSinogram randoms scatter] = OsemMmr(siemensSinogramFilename, span, overall_ncf_3d, attMapBaseFilename, randoms, scatter, outputPathIter, pixelSize_mm, numOsemSubsets, numOsemIterations, saveInterval, useGpu, stirMatlabPath, removeTempFiles);

    % Show slices:
    figure;
    imshow(volume(:,:,81)./max(max(volume(:,:,81))));
    title(sprintf('Reconstructed Image in Iteration %d', iteration));

    % 3) Project the reconstructed image:
    % Mask to remove odd pixels in the border:
    mask = volume > 0;
    mask = imerode(mask, strel('disk',7));
    volume = volume .* mask;
    [projectedImage, structSizeSinogram] = ProjectMmr(volume, pixelSize_mm, [outputPathIter pathBar sprintf('iter_%d/',iteration)], structSizeSino3d.span, 0,0, useGpu);

    % 4) Ratio between sinogram and projected:
    % Apply normalization and attenuation without crystal efficiencies:
    projectedImage_nf = projectedImage;
    projectedImage_nf = projectedImage_nf .*atteNormFactors + randoms + scatter .* overall_nf_3d;
    interfileWriteSino(projectedImage_nf, [outputPathIter pathBar sprintf('iter_%d/',iteration) pathBar 'ProjectedSino_Corrected'], structSizeSinogram);
    % 5) Estimate crystal efficiencies:    
    % Other method:
%     %a) Without mask:
%     useOfDetectorsInSinogram = detector1SystemMatrix'*double(sinogram(:)) + detector2SystemMatrix'*double(sinogram(:));
%     useOfDetectorsInProjection = detector1SystemMatrix'*double(projectedImage_nf(:)) + detector2SystemMatrix'*double(projectedImage_nf(:));
    % b) With mask:
    mask = volume > 0;
    mask = imerode(mask, strel('disk',7));
    meanVolume = mean(mean(mean(volume>0)));
    maskVolume = volume > meanVolume*0.7;
    [projectedMask, structSizeSinogram] = ProjectMmr(mask.*maskVolume, pixelSize_mm, [outputPathIter pathBar sprintf('iter_%d/mask/',iteration)], structSizeSino3d.span, 0,0, useGpu);
    projMask = projectedMask>0;
    useOfDetectorsInSinogram = detector1SystemMatrix'*double(sinogram(:).*projMask(:)) + detector2SystemMatrix'*double(sinogram(:).*projMask(:));
    useOfDetectorsInProjection = detector1SystemMatrix'*double(projectedImage_nf(:).*projMask(:)) + detector2SystemMatrix'*double(projectedImage_nf(:).*projMask(:));

    % Ratio of use of detectors:
    crystalEfficienciesCorrection = zeros(size(useOfDetectorsInProjection));
    crystalEfficienciesCorrection(useOfDetectorsInProjection~=0) = useOfDetectorsInSinogram(useOfDetectorsInProjection~=0)./useOfDetectorsInProjection(useOfDetectorsInProjection~=0);
    % Get spatially mean Value:
    lengthFilter = 24;
    meanValueCrystals = conv([ones(lengthFilter/2,1).*crystalEfficienciesCorrection(end); crystalEfficienciesCorrection(crystalEfficienciesCorrection~=0); ones(lengthFilter/2-1,1).*crystalEfficienciesCorrection(end)], ones(lengthFilter,1)./sum(ones(lengthFilter,1)), 'valid');
    crystalEfficienciesCorrection(crystalEfficienciesCorrection~=0) = crystalEfficienciesCorrection(crystalEfficienciesCorrection~=0)./meanValueCrystals;
    crystalEfficienciesCorrection(crystalEfficienciesCorrection~=0) = crystalEfficienciesCorrection(crystalEfficienciesCorrection~=0)./mean(crystalEfficienciesCorrection(crystalEfficienciesCorrection~=0));
    
    save([outputPath sprintf('crystalEfficienciesCorrection_iter%d', iteration)], 'crystalEfficienciesCorrection', '-v7.3' ); % Save crystal eff.
    % Is not a multiplicative factor, is the solution instead:
    %crystalEfficienciesVector = crystalEfficiencies(:) .* crystalEfficienciesCorrection;
    crystalEfficienciesVector = crystalEfficienciesCorrection;
    % 6) Save and compare qith the current crystal efficiencies:
    figure;
    plot(1:numDetectors, original_xtal_efficiencies(:), 1:numDetectors, crystalEfficienciesVector, 1:numDetectors, crystalEfficienciesCorrection);
    legend('Ground Truth', 'New Crystal Efficiencies', 'Crystal Efficiencies Correction');
%     % The crystal efficiences that were zeros because there were no counts,
%     % I left it with it's original value:
%     crystalEfficienciesVector(crystalEfficienciesVector == 0) = 1;
    % 7) Reshape crystal efficiences for the original format (by ring):
    crystalEfficiencies = reshape(crystalEfficienciesVector, [numDetectorsPerRing numRings]);
end

save([outputPath 'crystalEfficiencies_final'], 'crystalEfficiencies'); % Save crystal eff.