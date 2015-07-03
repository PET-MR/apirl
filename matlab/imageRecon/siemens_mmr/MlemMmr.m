%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 06/05/2015
%  *********************************************************************
%  This function reconstructs a mmr sinogram that can be in apirl interfile
%  or siemens interfile. It receives the
%  header of the sinogram, a normalization file, an attenuation map and the outputh
%  path for the results. It returns a volume.
% The filename for the attenuation map must include only the path and the
% first parte of the name as is stored by the mMR. For example:
% attMapBaseFilename = 'path/PET_ACQ_190_20150220152253', where the
% filenames are: 
%   - 'path/PET_ACQ_190_20150220152253_umap_human_00.v.hdr'
%   - 'path/PET_ACQ_190_20150220152253_umap_hardware_00.v.hdr'
%  It receives the span to be used in the reconstruction. The sinogram
%  It receives two optional parameters
%  - pixelSize_mm: that is a vector with 3 elements [pixelSizeX_mm
%  pixelSizeY_mm pixelSizeZ_mm].
%  - numIterations: number of iterations.
%
%  The span of the reconstruction is the same of the sinogram.
%
% Examples:
%   volume = MlemMmr(sinogramFilename, normFilename, attMapBaseFilename, outputPath, pixelSize_mm, numInputIterations, optUseGpu)
function volume = MlemMmr(sinogramFilename, normFilename, attMapBaseFilename, outputPath, pixelSize_mm, numInputIterations, optUseGpu)

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
numSubsets = 21;
numIterations = 3;
if nargin < 5
    % Default pixel size:
    pixelSize_mm = [2.08625 2.08625 2.03125];
    imageSize_pixels = [286 286 127]; 
    imageSize_mm = pixelSize_mm.*imageSize_pixels;
    useGpu = 0;
else
    imageSize_mm = [600 600 257];
    imageSize_pixels = ceil(imageSize_mm./pixelSize_mm);
    if nargin == 7
        numIterations = numInputIterations;
        useGpu = optUseGpu;
    else
        error('Wrong number of parameters: volume = OsemMmrSpan1(sinogramFilename, normFilename, attMapBaseFilename, outputPath, [2.08625 2.08625 2.03125], 60)');
        return;
    end
end


%% READING THE SINOGRAMS
disp('Read input sinogram...');
% Read the sinograms:
[sinograms, delayedSinograms, structSizeSino3d] = interfileReadSino(sinogramFilename);
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
disp('Computing the normalization correction factors...');
% ncf:
if ~isempty(normFilename)
    [overall_ncf_3d, scanner_time_invariant_ncf_3d, scanner_time_variant_ncf_3d, acquisition_dependant_ncf_3d, used_xtal_efficiencies, used_deadtimefactors, used_axial_factors] = ...
       create_norm_files_mmr(normFilename, [], [], [], [], structSizeSino3d.span);
    % invert for nf:
    overall_nf_3d = overall_ncf_3d;
    overall_nf_3d(overall_ncf_3d ~= 0) = 1./overall_nf_3d(overall_ncf_3d ~= 0);
else
    overall_ncf_3d = ones(size(sinogramSpan1));
    overall_nf_3d = overall_ncf_3d;
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
%% GENERATE AND SAVE ATTENUATION AND NORMALIZATION FACTORS AND CORECCTION FACTORS FOR SPAN11 SINOGRAMS FOR APIRL
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
%% GENERATE MLEM RECONSTRUCTION FILES FOR APIRL
if useGpu == 0
    disp('Mlem reconstruction...');
    saveInterval = 1;
    saveIntermediate = 0;
    outputFilenamePrefix = [outputPath 'reconImage'];
    filenameMlemConfig = [outputPath 'mlem.par'];
    CreateMlemConfigFileForMmr(filenameMlemConfig, [sinogramFilename '.h33'], [filenameInitialEstimate '.h33'], outputFilenamePrefix, numIterations, [],...
        saveInterval, saveIntermediate, [], [], [], [anfFilename '.h33']);
    % Execute APIRL: 
    status = system(['MLEM ' filenameMlemConfig]) 
else
    disp('cuMlem reconstruction...');
    saveInterval = 10;
    saveIntermediate = 0;
    outputFilenamePrefix = [outputPath 'reconImage'];
    filenameMlemConfig = [outputPath 'cumlem.par'];
    CreateCuMlemConfigFileForMmr(filenameMlemConfig, [sinogramFilename '.h33'], [filenameInitialEstimate '.h33'], outputFilenamePrefix, numIterations, [],...
        saveInterval, saveIntermediate, [], [], [], [anfFilename '.h33'], 0, 576, 576, 512);
    % Execute APIRL: 
    status = system(['cuMLEM ' filenameMlemConfig]) 
end

%% READ RESULTS
% Read interfile reconstructed image:
volume = interfileRead([outputFilenamePrefix '_final.h33']);