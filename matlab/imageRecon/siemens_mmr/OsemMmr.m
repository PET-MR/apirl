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
%  It receives three optional parameters
%  - pixelSize_mm: that is a vector with 3 elements [pixelSizeX_mm
%  pixelSizeY_mm pixelSizeZ_mm].
%  - numSubsets: number of subsets.
%  - numIterations: number of iterations.
%
%   -normFilename: can be a norm file o the overall_ncf matrix directly.
%
%  The span used in the reconstruction is received as a parameter.
%
% Examples:
%   volume = OsemMmrSpan1(sinogramFilename, normFilename, attMapBaseFilename, outputPath, [2.08625 2.08625 2.03125], 21, 3)
%   volume = OsemMmrSpan1(sinogramFilename, normFilename, attMapBaseFilename, outputPath, [2.08625 2.08625 2.03125])
%   volume = OsemMmrSpan1(sinogramFilename, normFilename, attMapBaseFilename, outputPath)
function volume = OsemMmr(sinogramFilename, span, normFilename, attMapBaseFilename, correctRandoms, correctScatter, outputPath, pixelSize_mm, numSubsets, numIterations, useGpu, stirMatlabPath)

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

% Check if we have received pixel size:
if nargin ~= 12
    error('Wrong number of parameters: volume = OsemMmr(sinogramFilename, span, normFilename, attMapBaseFilename, correctRandoms, correctScatter, outputPath, pixelSize_mm, numSubsets, numIterations, useGpu, stirMatlabPath)');
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
filenameInitialEstimate = [outputPath pathBar 'initialEstimate'];
interfilewrite(initialEstimate, filenameInitialEstimate, pixelSize_mm);
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
%% GENERATE AND SAVE ATTENUATION AND NORMALIZATION FACTORS AND CORECCTION FACTORS
disp('Generating the ANF sinogram...');
% Save:
outputSinogramName = [outputPath 'NF'];
interfileWriteSino(single(overall_nf_3d), outputSinogramName, structSizeSino3d);

% Compose with acfs:
atteNormFactors = overall_nf_3d;
atteNormFactors(acfsSinogram ~= 0) = overall_nf_3d(acfsSinogram ~= 0) ./acfsSinogram(acfsSinogram ~= 0);
anfFilename = [outputPath 'ANF'];
interfileWriteSino(single(atteNormFactors), anfFilename, structSizeSino3d);
clear atteNormFactors;
%clear normFactorsSpan11;
%% RANDOMS CORRECTION
randoms = zeros(size(sinograms));
if(correctRandoms)
    % Stir computes randoms that are already normalized:
    [randoms, structSizeSino] = estimateRandomsWithStir(delayedSinograms, structSizeSino3dSpan1, overall_ncf_3d, structSizeSino3d, [outputPath pathBar 'stirRandoms' pathBar]);
    % Precorrect the input sinogram because cuosem stil doesn't include the
    % additive sinogram:
    sinograms = sinograms - randoms;
    sinograms(sinograms < 0) = 0;
    % Rewrite input singoram:
    interfileWriteSino(single(sinograms), sinogramFilename, structSizeSino3d);
end
%% GENERATE MLEM RECONSTRUCTION FILES FOR APIRL
if useGpu == 0
    disp('Osem reconstruction...');
    saveInterval = 1;
    saveIntermediate = 0;
    outputFilenamePrefix = [outputPath 'reconImage'];
    filenameOsemConfig = [outputPath 'osem.par'];
    CreateOsemConfigFileForMmr(filenameOsemConfig, [sinogramFilename '.h33'], [filenameInitialEstimate '.h33'], outputFilenamePrefix, numIterations, [],...
        numSubsets, saveInterval, saveIntermediate, [], [], [], [anfFilename '.h33']);
    % Execute APIRL: 
    status = system(['OSEM ' filenameOsemConfig]) 
else
    disp('cuOsem reconstruction...');
    saveInterval = 10;
    saveIntermediate = 0;
    outputFilenamePrefix = [outputPath 'reconImage'];
    filenameMlemConfig = [outputPath 'cuosem.par'];
    CreateCuMlemConfigFileForMmr(filenameMlemConfig, [sinogramFilename '.h33'], [filenameInitialEstimate '.h33'], outputFilenamePrefix, numIterations, [],...
        saveInterval, saveIntermediate, [], [], [], [anfFilename '.h33'], 0, 576, 576, 512, numSubsets);
    % Execute APIRL: 
    status = system(['cuMLEM ' filenameMlemConfig]) 
end
%% READ RESULTS
% Read interfile reconstructed image:
volume = interfileRead([outputFilenamePrefix '_final.h33']);

%% IF NEEDS TO CORRECT SCATTER, ESTIMATE IT AFTER RECONSTRUCTION:
if correctScatter
    thresholdForTail = 1.01;
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

    % The emission sinogram needs to be normalized and corrected for randoms:
    outputPathScatter1 = [outputPath pathBar 'Scatter1' pathBar];
    sinogram_precorrected = sinograms.*overall_ncf_3d;  % Already substracted randoms.
    sinogram_precorrected(sinogram_precorrected<0) = 0;
    [scatter_1, structSizeSino, mask] = estimateScatterWithStir(volume, attenMap, pixelSize_mm, sinogram_precorrected, acfsOnlyHuman, structSizeSino3d, outputPathScatter1, stirScriptsPath, thresholdForTail);
    
    % Reconstruct again with the scatter:
    interfileWriteSino(single(scatter_1), [outputPathScatter1 'scatter'], structSizeSino3d);
    
    sino_precorrected = (sinograms - randoms).*overall_ncf_3d;
    %sino_precorrected(sino_precorrected<0) = 0; % Avoid this for the
    %scaling
    for sinogram_to_scale = 1:size(scatter_1,3)  
        maskNonZeros = overall_ncf_3d(:,:,sinogram_to_scale) ~= 0;
    scat_2d_sino = scatter_1(:,:,sinogram_to_scale).*maskNonZeros;
    
    true_2d_sino = sino_precorrected(:,:,sinogram_to_scale);
    acfs_2d_sino = acfsOnlyHuman(:,:,sinogram_to_scale);
    outside2d_indices(:,:,sinogram_to_scale) = acfs_2d_sino <= 1.01; % Regions
    

    scale_factor = sum(true_2d_sino(outside2d_indices(:,:,sinogram_to_scale))) / sum(scat_2d_sino(outside2d_indices(:,:,sinogram_to_scale)))
    %scan_2d_sino = scatter_with_norm_problem(:,:,sinogram_to_scale);

    scatter_1_scaled(:,:,sinogram_to_scale) = scatter_1(:,:,sinogram_to_scale) .* scale_factor;
    end

    profileSinogram = sum(sino_precorrected(:,126,:),3);
    profileScatter = sum(scatter_1(:,126,:),3);
    profileScatter_scaled = sum(scatter_1_scaled(:,126,:),3);
    figure;
    plot([profileSinogram profileScatter profileScatter_scaled]);
    
    normScatter = scatter_1_scaled .* overall_nf_3d;
profileSinogram = sum(sinograms(:,126,:),3);
profileRandoms = sum(randoms(:,126,:),3);
profileNormScatter = sum(normScatter(:,126,:),3);
figure;
plot([profileSinogram profileRandoms profileNormScatter (profileRandoms+profileNormScatter)]);
legend('Sinogram', 'Randoms', 'Scatter', 'Randoms+Scatter');

    sinogram_scatter_corrected = sinograms  - randoms - scatter_1 .* overall_nf_3d;
    sinogram_scatter_corrected(sinogram_scatter_corrected<0) = 0;
    % Rewrite input singoram:
    sinogramFilename = [outputPathScatter1 pathBar 'sinogram'];
    interfileWriteSino(single(sinogram_scatter_corrected), sinogramFilename, structSizeSino3d);
    if useGpu == 0
        disp('Osem reconstruction...');
        saveInterval = 1;
        saveIntermediate = 0;
        outputFilenamePrefix = [outputPathScatter1 'reconImage'];
        filenameOsemConfig = [outputPathScatter1 'osem.par'];
        CreateOsemConfigFileForMmr(filenameOsemConfig, [sinogramFilename '.h33'], [filenameInitialEstimate '.h33'], outputFilenamePrefix, numIterations, [],...
            numSubsets, saveInterval, saveIntermediate, [], [], [], [anfFilename '.h33']);
        % Execute APIRL: 
        status = system(['OSEM ' filenameOsemConfig]) 
    else
        disp('cuOsem reconstruction...');
        saveInterval = 10;
        saveIntermediate = 0;
        outputFilenamePrefix = [outputPathScatter1 'reconImage'];
        filenameMlemConfig = [outputPathScatter1 'cuosem.par'];
        CreateCuMlemConfigFileForMmr(filenameMlemConfig, [sinogramFilename '.h33'], [filenameInitialEstimate '.h33'], outputFilenamePrefix, numIterations, [],...
            saveInterval, saveIntermediate, [], [], [], [anfFilename '.h33'], 0, 576, 576, 512, numSubsets);
        % Execute APIRL: 
        status = system(['cuMLEM ' filenameMlemConfig]) 
    end
    volume = interfileRead([outputFilenamePrefix '_final.h33']);

    
    outputPathScatter2 = [outputPath pathBar 'Scatter2' pathBar];
    [scatter_2, structSizeSino, mask] = estimateScatterWithStir(volume, attenMap, pixelSize_mm, sinogram_precorrected, acfsOnlyHuman, structSizeSino3d, outputPathScatter2, stirScriptsPath, thresholdForTail);
    interfileWriteSino(single(scatter_2), [outputPathScatter2 'scatter'], structSizeSino3d);
    scatter = (scatter_1+scatter_2)./2;
    interfileWriteSino(single(scatter), [outputPathScatter2 'scatter_1_2'], structSizeSino3d);
    % Reconstruct again with the scatter:
    sinogram_scatter_corrected = sinograms - scatter .* overall_nf_3d;
    % Rewrite input singoram:
    sinogramFilename = [outputPathScatter2 pathBar 'sinogram'];
    interfileWriteSino(single(sinogram_scatter_corrected), sinogramFilename, structSizeSino3d);
    if useGpu == 0
        disp('Osem reconstruction...');
        saveInterval = 1;
        saveIntermediate = 0;
        outputFilenamePrefix = [outputPathScatter2 'reconImage'];
        filenameOsemConfig = [outputPathScatter2 'osem.par'];
        CreateOsemConfigFileForMmr(filenameOsemConfig, [sinogramFilename '.h33'], [filenameInitialEstimate '.h33'], outputFilenamePrefix, numIterations, [],...
            numSubsets, saveInterval, saveIntermediate, [], [], [], [anfFilename '.h33']);
        % Execute APIRL: 
        status = system(['OSEM ' filenameOsemConfig]) 
    else
        disp('cuOsem reconstruction...');
        saveInterval = 10;
        saveIntermediate = 0;
        outputFilenamePrefix = [outputPathScatter2 'reconImage'];
        filenameMlemConfig = [outputPathScatter2 'cuosem.par'];
        CreateCuMlemConfigFileForMmr(filenameMlemConfig, [sinogramFilename '.h33'], [filenameInitialEstimate '.h33'], outputFilenamePrefix, numIterations, [],...
            saveInterval, saveIntermediate, [], [], [], [anfFilename '.h33'], 0, 576, 576, 512, numSubsets);
        % Execute APIRL: 
        status = system(['cuMLEM ' filenameMlemConfig]) 
    end
    volume = interfileRead([outputFilenamePrefix '_final.h33']);
end