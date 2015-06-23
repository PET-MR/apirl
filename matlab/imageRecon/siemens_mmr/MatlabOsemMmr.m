%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 06/05/2015
%  *********************************************************************
%  This function reconstructs a mmr span n sinogram. It receives the
%  uncompressed sinogram raw data, a normalization file, an attenuation map and the outputh
%  path for the results. It returns a volume.
% The filename for the attenuation map must include only the path and the
% first parte of the name as is stored by the mMR. For example:
% attMapBaseFilename = 'path/PET_ACQ_190_20150220152253', where the
% filenames are: 
%   - 'path/PET_ACQ_190_20150220152253_umap_human_00.v.hdr'
%   - 'path/PET_ACQ_190_20150220152253_umap_hardware_00.v.hdr'
%
%  It receives two optional parameters
%  - pixelSize_mm: that is a vector with 3 elements [pixelSizeX_mm
%  pixelSizeY_mm pixelSizeZ_mm].
%  - numIterations: number of iterations.
%
%  To run the fastest version, use MlemMmr that runs the binary APIRL
%  version directly. This version runs with APIRL only the projection and
%  backprojection. Stores each iteration in a different folder from the bas
%  outputPath.
%  
%  The span for the reconstruction is RECEIVED AS A PARAMETER.
%
%  The save interval parameter permits to store data of each "saveInterval"
%  iterations, if zero, only temporary files are written in a temp folder
%  and are overwritten for each iteration.
% Examples:
%   [volume randoms scatter] = MatlabOsemMmr(sinogramFilename, span, normFilename, attMapBaseFilename, correctRandoms, correctScatter, outputPath, pixelSize_mm, numSubsets, numIterations, saveInterval, useGpu, stirMatlabPath)

function [volume randoms scatter] = MatlabOsemMmr(sinogramFilename, span, normFilename, attMapBaseFilename, correctRandoms, correctScatter, outputPath, pixelSize_mm, numSubsets, numIterations, saveInterval, useGpu, stirMatlabPath)

if ~isdir(outputPath)
    mkdir(outputPath);
end

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

% Check if we have received pixel size:
if nargin ~= 13
    error('Wrong number of parameters: [volume randoms scatter] = MatlabOsemMmr(sinogramFilename, span, normFilename, attMapBaseFilename, correctRandoms, correctScatter, outputPath, pixelSize_mm, numSubsets, numIterations, saveInterval, useGpu, stirMatlabPath)');
end
imageSize_mm = [600 600 257.96875];
imageSize_pixels = ceil(imageSize_mm./pixelSize_mm);

%% READING THE SINOGRAMS
disp('Read input sinogram...');
% Read the sinograms:
[sinograms, delayedSinograms, structSizeSino3dSpan1] = interfileReadSino(sinogramFilename);
% Convert to span:
[sinograms, structSizeSino3d] = convertSinogramToSpan(sinograms, structSizeSino3dSpan1, span);
sinogramFilename = [outputPath pathBar 'sinogram'];
% Write the input sinogram:
interfileWriteSino(single(sinograms), sinogramFilename, structSizeSino3d);
%% CREATE INITIAL ESTIMATE FOR RECONSTRUCTION
disp('Creating inital image...');
% Inititial estimate:
initialEstimate = ones(imageSize_pixels, 'single');
%filenameInitialEstimate = [outputPath pathBar 'initialEstimate'];
%interfilewrite(initialEstimate, filenameInitialEstimate, pixelSize_mm);
%% NORMALIZATION FACTORS
if isstr(normFilename)
    disp('Computing the normalization correction factors...');
    % ncf:
    [overall_ncf_3d, scanner_time_invariant_ncf_3d, scanner_time_variant_ncf_3d, used_xtal_efficiencies, used_deadtimefactors, used_axial_factors] = ...
       create_norm_files_mmr(normFilename, [], [], [], [], structSizeSino3d.span);
    % invert for nf:
    overall_nf_3d = overall_ncf_3d;
    overall_nf_3d(overall_ncf_3d ~= 0) = 1./overall_nf_3d(overall_ncf_3d ~= 0);
else
    if size(normFilename) ~= size(sinogramSpan1)
        error('The size of the normalization correction factors is incorrect.')
    end
    disp('Using the normalization correction factors received as a parameter...');
    overall_ncf_3d = normFilename;
    clear normFilename;
    % invert for nf:
    overall_nf_3d = overall_ncf_3d;
    overall_nf_3d(overall_ncf_3d ~= 0) = 1./overall_nf_3d(overall_ncf_3d ~= 0);
end
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
%% RANDOM ESTIMATE
% If the edlayed sinograms are available and stir is availables, use it, if
% not use the singles per bucket.
% Check if we hve the randoms estimate as a parameter. If not we compute it
% at the end. I will save in randoms the randoms estimate, and in
% randomsCorrection I will overwrite it to use it as a flag.
% Stir computes randoms that are already normalized:
randoms = zeros(size(sinograms));
if numel(size(correctRandoms)) == numel(size(sinograms))
    if size(correctRandoms) == size(sinograms)
        % The input is the random estimate:
        randoms = correctRandoms;
        % Precorrect the input sinogram because cuosem stil doesn't include the
        % additive sinogram:
        sinograms_rand_scat_subtracted = sinograms - randoms;
        sinograms_rand_scat_subtracted(sinograms_rand_scat_subtracted < 0) = 0;
        % Rewrite input singoram:
        interfileWriteSino(single(sinograms_rand_scat_subtracted), sinogramFilename, structSizeSino3d);
    else
        randoms = zeros(size(sinograms));
    end
else
    % If not, we expect a 1 to estimate randoms or a 0 to not cprrect for
    % them:
    if(correctRandoms)
        % Stir computes randoms that are already normalized:
        [randoms, structSizeSino] = estimateRandomsWithStir(delayedSinograms, structSizeSino3dSpan1, overall_ncf_3d, structSizeSino3d, [outputPath pathBar 'stirRandoms' pathBar]);
    else
        randoms = zeros(size(sinograms));
    end
end
%% SCATTER CORRECTION
% Check if we hve the scatter estimate as a parameter. If not we compute it
% at the end. I will save in scatter the scatter estimate, and in
% computeScatter we enable to estimate the scatter.
if numel(size(correctScatter)) == numel(size(sinograms))
    if size(correctScatter) == size(sinograms)
        scatter = correctScatter;
        computeScatter = 0;    % Its not necesary to compute scatter correction because I already have the estimate.
    else
        scatter = zeros(size(sinograms));
        computeScatter = 1;    % It's necessary to estimate scatter correction.
    end
else
    scatter = zeros(size(sinograms));
    if (numel(correctScatter) == 1) 
        if (correctScatter == 0)
            computeScatter = 0; % Not correct for scatter only when correctScatter=0
        else
            computeScatter = 1; % It's necessary to estimate scatter correction.
        end
    else
        computeScatter = 1; % It's necessary to estimate scatter correction.
    end
end
% If I need to compute the scatter, I need the acfs of only the human
% attenuation map:
if computeScatter == 1
    stirScriptsPath = [stirMatlabPath pathBar 'scripts'];
    % The scatter needs the image but also the acf to scale, and in the case of
    % the mr is better if this acf include the human?
    if ~strcmp(attMapBaseFilename(end-3:end),'.h33')
        acfFilename = 'acfsOnlyHuman';
        acfsOnlyHuman = createACFsFromImage(attenMap_human, imageSizeAtten_mm, outputPath, acfFilename, sinogramFilename, structSizeSino3d, 0, useGpu);
        acfFilename = [outputPath acfFilename];
    else
        acfsOnlyHuman = acfsSinogram;
    end
end
%% MLEM
% Runs ordinary poission mlem.
disp('###################### OS-EM RECONSTRUCTION ##########################');
% 1) Compute sensitivity image. Backprojection of attenuation and
% normalization factors:
disp('Compute sensitivity images...');
anfSino = overall_ncf_3d .* acfsSinogram; % ancf.
anfSino(anfSino~= 0) = 1./anfSino(anfSino~= 0); % anf.
% One sensitivity image per path:
for s = 1 : numSubsets
    % Backproject sinogram in other path:
    sensitivityPath = [outputPath sprintf('SensitivityImage_%d', s) pathBar];
    mkdir(sensitivityPath);
    [sensitivityImages(:,:,:,s), pixelSize_mm] = BackprojectMmr(anfSino, imageSize_pixels, pixelSize_mm, sensitivityPath, structSizeSino3d.span, numSubsets,s, useGpu);
    % Generate update threshold:
    updateThreshold(s) =  min(min(min(sensitivityImages(:,:,:,s))))+ ( max(max(max(sensitivityImages(:,:,:,s))))- min(min(min(sensitivityImages(:,:,:,s))))) * 0.001;
end

% Scatter parameters:
thresholdForTail = 1.01;
stirScriptsPath = [stirMatlabPath pathBar 'scripts'];
% 2) OSEM Reconstruction.
if computeScatter == 0
    numItersScatter = 1;
else
    numItersScatter = 3;
end
scatter = zeros(size(sinograms));
for iterScatter = 1 : numItersScatter
    emRecon = initialEstimate; % em_recon is the current reconstructed image.
    for iter = 1 : numIterations
        for s = 1 : numSubsets
            % Sens image:
            sensImage = sensitivityImages(:,:,:,s);

            % I work with a sinogram of the original size, instead of the
            % subset. Project generates a sinogram of the same size of the
            % orignal size, but only fills the bins of the subset.
    %         % Get subset of the input sinogram and the anfs:
    %         [inputSubset, structSizeSino3dSubset] = getSubsetFromSinogram(sinograms, structSizeSino3d, numSubsets, s);
    %         [anfSubset, structSizeSino3dSubset] = getSubsetFromSinogram(anfSino, structSizeSino3d, numSubsets, s);

            disp(sprintf('Iteration %d...', iter));
            % 2.a) Create working directory:
            if rem(iter,saveInterval) == 0
                iterationPath = [outputPath pathBar sprintf('Iteration%d', iter) pathBar];
                if ~isdir(iterationPath) 
                    mkdir(iterationPath);
                end
            else
                iterationPath = [outputPath 'temp' pathBar];
                if ~isdir(iterationPath) 
                    mkdir(iterationPath);
                end
            end
            % 2.b) Project current image:
            [projectedImage, structSizeSinogram] = ProjectMmr(emRecon, pixelSize_mm, iterationPath, structSizeSino3d.span, numSubsets,s, useGpu);
            % 2.c) Multiply by the anf and add randoms and scatter:
            maskSubset = projectedImage ~= 0; % Matrix with zeros in the bins that are not of the subset.
            projectedImage = projectedImage .* anfSino + maskSubset.*randoms + maskSubset.*scatter .* overall_nf_3d;  % Randoms are already normalized, but scatter is not.
            % 2.d) Divide sinogram by projected sinogram:
            ratioSinograms = zeros(size(projectedImage), 'single');
            ratioSinograms(projectedImage~=0) = sinograms(projectedImage~=0) ./ projectedImage(projectedImage~=0);
            ratioSinograms = ratioSinograms.* anfSino; % Apply anf.
            % 2.e) Backprojection:
            [backprojImage, pixelSize_mm] = BackprojectMmr(ratioSinograms, imageSize_pixels, pixelSize_mm, iterationPath, structSizeSino3d.span, numSubsets,s, useGpu);
            % 2.f) Apply sensitiivty image and correct current image:
            emRecon(sensImage > updateThreshold(s)) = emRecon(sensImage > updateThreshold(s)) .* backprojImage(sensImage > updateThreshold(s))./ sensImage(sensImage > updateThreshold(s));
            emRecon(sensImage <= updateThreshold(s)) = 0;
        end
        if rem(iter,saveInterval) == 0
            interfilewrite(emRecon, [outputPath pathBar sprintf('emImage_iter%d', iter)], pixelSize_mm);
            show_sinos(emRecon, 3, 'Recon Image', 1 );
        end
    end
    if iterScatter < numItersScatter    % For the last iteration is not necessary to estimate the scatter, because the scatter is used in the next iteration.
        % Correct scatter in fact is a flag that estim
        if computeScatter
            % SCATTER ESTIMATE
            % It also uses stir:
            outputPathScatter = [outputPath pathBar sprintf('scatter_%d', iterScatter) pathBar];
            [scatterEstimates{iterScatter}, structSizeSino, mask] = estimateScatterWithStir(emRecon, attenMap, pixelSize_mm, sinograms, randoms, overall_ncf_3d, acfsOnlyHuman, structSizeSino3d, outputPathScatter, stirScriptsPath, thresholdForTail);
            % Save old scatter to average them.
            % Average:
            scatter = zeros(size(sinograms));
            for i = 1 : iterScatter
                scatter = scatter + scatterEstimates{iterScatter};
            end
            scatter = scatter ./ iterScatter;
            interfileWriteSino(single(scatter), [outputPath pathBar sprintf('scatterEstimate_iter%d', iterScatter)], structSizeSino);
        end
    end
end
%% OUTPUT PARAMETER
interfilewrite(emRecon, [outputPath pathBar 'emImage_final'], pixelSize_mm);
volume = emRecon;