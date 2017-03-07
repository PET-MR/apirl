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


function [volume, crystalEfficiencies] = SelfNormRecon(sinogramFilename, normFilename, attMapBaseFilename, correctRandoms, correctScatter, outputPath, pixelSize_mm, numOsemSubsets, numOsemIterations, numXtalIterations, filenameSystemMatrix1, filenameSystemMatrix2, stirMatlabPath, applyNormMask)

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
if isstr(sinogramFilename)
    [sinograms, delayedSinograms, structSizeSino3dSpan1] = interfileReadSino(sinogramFilename);
    if structSizeSino3dSpan1.span == 1
        % Convert to span:
        [sinograms, structSizeSino3d] = convertSinogramToSpan(sinograms, structSizeSino3dSpan1, span);  
    else
        structSizeSino3d = structSizeSino3dSpan1;
        warning('The span parameter will be ignored because the input sinogram is not span 1.');
    end
else
    % binary sinogram (span-1?) or span span
    if size(sinogramFilename) == [344 252 4084]
        structSizeSino3d = getSizeSino3dFromSpan(344, 252, 64, ...
            296, 256, 1, 60);
        sinograms = sinogramFilename;
    else
        structSizeSino3d = getSizeSino3dFromSpan(344, 252, 64, ...
            296, 256, span, 60);
        if size(sinogramFilename) == [344 252 sum(structSizeSino3d.sinogramsPerSegment)]
            sinograms = sinogramFilename;
        else
            error('Invalid sinogram matrix size. It must be span 1 or "span".');
        end
    end
end
sinogramFilename = [outputPath pathBar 'sinogram'];
% Write the input sinogram:
interfileWriteSino(single(sinograms), sinogramFilename, structSizeSino3d);
%% PROCESS INPUT PARAMETERS
% Image size:
imageSize_mm = [600 600 258];
imageSize_pixels = ceil(imageSize_mm./pixelSize_mm);

if isempty(filenameSystemMatrix1) || isempty(filenameSystemMatrix2)
    disp('Computing detector system matrix...');
    [detector1SystemMatrix, detector2SystemMatrix] = createDetectorSystemMatrix3d(structSizeSino3d.span, 1);
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
%% ATTENUATION FACTORS
% Norm factors are generated in each iteration during reconstruction.
if isstr(attMapBaseFilename)
    % Check if its attenuation from siemens or a post processed image:
    if ~strcmp(attMapBaseFilename(end-3:end),'.h33')
        disp('Computing the attenuation correction factors from mMR mu maps...');
        % Read the attenuation map and compute the acfs.
        % Check if we have the extended version for the mumaps:
        if exist([attMapBaseFilename '_umap_human_ext_000_000_00.v.hdr'])
            attMapHumanFilename = [attMapBaseFilename '_umap_human_ext_000_000_00.v.hdr'];
        else
            attMapHumanFilename = [attMapBaseFilename '_umap_human_00.v.hdr'];
        end
        [attenMap_human, refAttenMapHum, bedPosition_mm, info]  = interfileReadSiemensImage(attMapHumanFilename);
        [attenMap_hardware, refAttenMapHard, bedPosition_mm, info]  = interfileReadSiemensImage([attMapBaseFilename '_umap_hardware_00.v.hdr']);
        imageSizeAtten_mm = [refAttenMapHum.PixelExtentInWorldY refAttenMapHum.PixelExtentInWorldX refAttenMapHum.PixelExtentInWorldZ];
        % Compose both images:
        attenMap = attenMap_hardware + attenMap_human;
        % I need to translate because siemens uses an slightly displaced
        % center (taken from dicom images, the first pixel is -359.8493 ,-356.8832 
        %displacement_mm = [-1.5 -imageSizeAtten_mm(2)*size(attenMap_human,1)/2+356.8832 0];
        %[attenMap, Rtranslated] = imtranslate(attenMap, refAttenMapHum, displacement_mm,'OutputView','same');
        
    else
        disp('Computing the attenuation correction factors from post processed APIRL mu maps...');
        attenMap_human = interfileRead(attMapBaseFilename);
        attenMap = attenMap_human;
        infoAtten = interfileinfo(attMapBaseFilename); 
        imageSizeAtten_mm = [infoAtten.ScalingFactorMmPixel1 infoAtten.ScalingFactorMmPixel2 infoAtten.ScalingFactorMmPixel3];
    end   

    % Create ACFs of a computed phatoms with the linear attenuation
    % coefficients:
    acfFilename = ['acfsSinogram'];
    acfsSinogram = createACFsFromImage(attenMap, imageSizeAtten_mm, outputPath, acfFilename, structSizeSino3d, 0, useGpu);

    % After the projection read the acfs:
    acfFilename = [outputPath acfFilename];
    fid = fopen([acfFilename '.i33'], 'r');
    numSinos = sum(structSizeSino3d.sinogramsPerSegment);
    [acfsSinogram, count] = fread(fid, structSizeSino3d.numTheta*structSizeSino3d.numR*numSinos, 'single=>single');
    acfsSinogram = reshape(acfsSinogram, [structSizeSino3d.numR structSizeSino3d.numTheta numSinos]);
    % Close the file:
    fclose(fid);
else
    if isempty(attMapBaseFilename)
        acfFilename = '';
        acfsSinogram = ones(size(sinograms));
    elseif size(attMapBaseFilename) == size(sinograms)
        % I got the acfs:
        acfsSinogram = attMapBaseFilename;
    end
end
%clear normFactorsSpan11;
%% RANDOMS CORRECTION
randoms = zeros(size(sinograms));
if numel(size(correctRandoms)) == numel(size(sinograms))
    if size(correctRandoms) == size(sinograms)
        % The input is the random estimate:
        randoms = correctRandoms;
    else
        [randoms, structSizeSino] = estimateRandomsWithStir(delayedSinogramSpan1, structSizeSino3dSpan1, overall_ncf_3d, structSizeSino3d, [outputPath pathBar 'stirRandoms' pathBar]);
    end
else
    % If not, we expect a 1 to estimate randoms or a 0 to not cprrect for
    % them:
    if(correctRandoms)
        % Stir computes randoms that are already normalized:
        numRandomsIter = 3;
        [randoms, structSizeSino] = estimateRandomsFromDelayeds(delayedSinogramSpan1, structSizeSino3dSpan1, numRandomsIter, structSizeSino3dSpan1.span);
    end
end
%% SCATTER CORRECTION
% Check if we hve the scatter estimate as a parameter. If not we compute it
% at the end:
scatter = zeros(size(sinograms));
if numel(size(correctScatter)) == numel(size(sinograms))
    if size(correctScatter) == size(sinograms)
        scatter = correctScatter;
    end
end
%% SELF NORMALIZING ALGORITHM

% Initialize crystal efficiencies:
numDetectorsPerRing = 504;
numRings = 64;
numDetectors = numDetectorsPerRing*numRings;
crystalEfficiencies = ones(numDetectorsPerRing, numRings); % Initialize the crystal efficiencies with ones.
% crystalEfficiencies = 2.* rand(size(original_xtal_efficiencies));
% crystalEfficiencies(original_xtal_efficiencies == 0) = 0;
saveInterval = 1;
useGpu = 1;
removeTempFiles = 1;

% 1) Compute normalization factors with the crystal efficiencies:
    [overall_ncf_3d, scanner_time_invariant_ncf_3d, scanner_time_variant_ncf_3d, acquisition_dependant_ncf_3d, xtal_dependant_ncf_3d, crystalEfficiencies, used_deadtimefactors, used_axial_factors] = ...
        create_norm_files_mmr(normFilename, [], crystalEfficiencies, [], [], structSizeSino3d.span);  % Recompute normalization factors with the crystal efficiencies.
    
disp('########## STARTING ITERATIVE ALGORITHM ##########');
% Iterative process:
for iteration = 1 : numXtalIterations
    outputPathIter = [outputPath sprintf('Iteration_%d/', iteration)];
    if ~isdir(outputPathIter)
        mkdir(outputPathIter)
    end
    disp(sprintf('Iteration No %d', iteration));
    disp(sprintf('Compute NCF in iteration %d', iteration));
    
    save([outputPathIter sprintf('crystalEfficiencies_iter%d', iteration)], 'crystalEfficiencies'); % Save crystal eff.
    crystalEffCorrFactorsFilename = [outputPathIter sprintf('CrystalEffCorrFactors_iter%d', iteration)]; % Save crystal correction factors.

    [volume overall_ncf_3d_osem acfsSinogram randoms scatter] = OsemMmr(sinograms, structSizeSino3d.span, crystalEfficiencies, attMapBaseFilename, randoms, scatter, outputPathIter, pixelSize_mm, numOsemSubsets, numOsemIterations, saveInterval, useGpu, stirMatlabPath, removeTempFiles);

    % Show slices:
    figure;
    imshow(volume(:,:,81)./max(max(volume(:,:,81))));
    title(sprintf('Reconstructed Image in Iteration %d', iteration));
    
    % 3) Project the reconstructed image:
    % Mask to remove odd pixels in the border:
    mask = volume > 0;
    mask = imerode(mask, strel('disk',7));
    volume = volume .* mask;
    [projectedImage, structSizeSinogram] = ProjectMmr(volume, pixelSize_mm, [outputPathIter pathBar sprintf('temp_projection/',iteration)], structSizeSino3d.span, 0,0, useGpu);
    % 4) Ratio between sinogram and projected:
    % Apply normalization and attenuation without crystal efficiencies:
%     overall_nf_3d = overall_ncf_3d;
%     overall_nf_3d(overall_nf_3d~=0) = 1./(overall_nf_3d(overall_nf_3d~=0));
    xtal_dependant_nf_3d = xtal_dependant_ncf_3d;
    xtal_dependant_nf_3d(xtal_dependant_nf_3d~=0) = 1./(xtal_dependant_nf_3d(xtal_dependant_nf_3d~=0));
    atteNormFactors = acfsSinogram .* overall_ncf_3d;
    atteNormFactors(atteNormFactors~=0) = 1./(atteNormFactors(atteNormFactors~=0));
    atteNormFactors(atteNormFactors==0) = 0;
    projectedImage_nf = projectedImage;
    projectedImage_nf = projectedImage_nf .*atteNormFactors + randoms + scatter .* xtal_dependant_nf_3d; % additive = (randoms + scatter.* overall_nf_3d_scatter);
    interfileWriteSino(projectedImage_nf, [outputPathIter pathBar 'ProjectedSino_Corrected'], structSizeSinogram);
    
    % Compute likelihood:
    log_likelihood(iteration) = sum(sinograms(:).*log(projectedImage(:))-projectedImage(:));
    
    % 5) Estimate crystal efficiencies:    
    % Other method:
%     %a) Without mask:
%     useOfDetectorsInSinogram = detector1SystemMatrix'*double(sinogram(:)) + detector2SystemMatrix'*double(sinogram(:));
%     useOfDetectorsInProjection = detector1SystemMatrix'*double(projectedImage_nf(:)) + detector2SystemMatrix'*double(projectedImage_nf(:));
    % b) With mask:
    if applyNormMask == 0
        %useOfDetectorsInSinogram = detector1SystemMatrix'*double(sinograms(:)) + detector2SystemMatrix'*double(sinograms(:));
        %useOfDetectorsInProjection = detector1SystemMatrix'*double(projectedImage_nf(:)) + detector2SystemMatrix'*double(projectedImage_nf(:));
        useOfDetectorsInSinogram = detector1SystemMatrix'*double(sinograms(:)) + detector2SystemMatrix'*double(sinograms(:));
        useOfDetectorsInProjection = detector1SystemMatrix'*(double(projectedImage_nf(:)).*(detector2SystemMatrix*double(crystalEfficiencies(:)))) + detector2SystemMatrix'*(double(projectedImage_nf(:)).*(detector1SystemMatrix*double(crystalEfficiencies(:))));
    else
        mask = volume > 0;
        mask = imerode(mask, strel('disk',7));
        meanVolume = mean(mean(mean(volume>0)));
        maskVolume = volume > meanVolume*0.7;
        [projectedMask, structSizeSinogram] = ProjectMmr(mask.*maskVolume, pixelSize_mm, [outputPathIter pathBar sprintf('temp_projection/mask/',iteration)], structSizeSino3d.span, 0,0, useGpu);
        projMask = projectedMask>0;
        useOfDetectorsInSinogram = detector1SystemMatrix'*double(sinograms(:).*projMask(:)) + detector2SystemMatrix'*double(sinograms(:).*projMask(:));
        useOfDetectorsInProjection = detector1SystemMatrix'*double(projectedImage_nf(:).*projMask(:)) + detector2SystemMatrix'*double(projectedImage_nf(:).*projMask(:));
    end
    % Ratio of use of detectors:
    crystalEfficienciesCorrection = zeros(size(useOfDetectorsInProjection));
    crystalEfficienciesCorrection(useOfDetectorsInProjection~=0) = useOfDetectorsInSinogram(useOfDetectorsInProjection~=0)./useOfDetectorsInProjection(useOfDetectorsInProjection~=0);
    
    % When using the mask (real data), also filter the crystal signal:
    if applyNormMask ~= 0
        % Get spatially mean Value:
        lengthFilter = 24;
        meanValueCrystals = conv([ones(lengthFilter/2,1).*crystalEfficienciesCorrection(end); crystalEfficienciesCorrection(crystalEfficienciesCorrection~=0); ones(lengthFilter/2-1,1).*crystalEfficienciesCorrection(end)], ones(lengthFilter,1)./sum(ones(lengthFilter,1)), 'valid');
        crystalEfficienciesCorrection(crystalEfficienciesCorrection~=0) = crystalEfficienciesCorrection(crystalEfficienciesCorrection~=0)./meanValueCrystals;
        crystalEfficienciesCorrection(crystalEfficienciesCorrection~=0) = crystalEfficienciesCorrection(crystalEfficienciesCorrection~=0)./mean(crystalEfficienciesCorrection(crystalEfficienciesCorrection~=0));
    end
    
    save([outputPath sprintf('crystalEfficienciesCorrection_iter%d', iteration)], 'crystalEfficienciesCorrection', '-v7.3' ); % Save crystal eff.
    % Is not a multiplicative factor, is the solution instead:
    crystalEfficienciesVector = crystalEfficienciesCorrection(:);% .* crystalEfficienciesCorrection;
    % 6) Save and compare qith the current crystal efficiencies:
%     figure;
%     plot(1:numDetectors, original_xtal_efficiencies(:), 1:numDetectors, crystalEfficienciesVector, 1:numDetectors, crystalEfficienciesCorrection);
%     legend('Ground Truth', 'New Crystal Efficiencies', 'Crystal Efficiencies Correction');
%     % The crystal efficiences that were zeros because there were no counts,
%     % I left it with it's original value:
%     crystalEfficienciesVector(crystalEfficienciesVector == 0) = 1;
    % 7) Reshape crystal efficiences for the original format (by ring):
    crystalEfficiencies = reshape(crystalEfficienciesVector, [numDetectorsPerRing numRings]);
end

save([outputPath 'crystalEfficiencies_final'], 'crystalEfficiencies'); % Save crystal eff.
save([outputPath 'log_likelihood'], 'log_likelihood');