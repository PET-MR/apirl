%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 06/05/2015
%  *********************************************************************
%  This function reconstructs a mmr span 1 sinogram. It receives the
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
% Examples:
%   volume = OsemMmrSpan1(sinogramFilename, normFilename, attMapBaseFilename, outputPath, [2.08625 2.08625 2.03125], 21, 3)
%   volume = OsemMmrSpan1(sinogramFilename, normFilename, attMapBaseFilename, outputPath, [2.08625 2.08625 2.03125])
%   volume = OsemMmrSpan1(sinogramFilename, normFilename, attMapBaseFilename, outputPath)
function volume = OsemMmrSpan1(sinogramFilename, normFilename, attMapBaseFilename, outputPath, pixelSize_mm, numInputSubsets, numInputIterations)

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
    imageSize_mm = pixelSize_mm.*imageSize_pixelsl
else
    imageSize_mm = [600 600 258];
    imageSize_pixels = ceil(imageSize_mm./pixelSize_mm);
    if nargin == 7
        numSubsets = numInputSubsets;
        numIterations = numInputIterations;
    else
        error('Wrong number of parameters: volume = OsemMmrSpan1(sinogramFilename, normFilename, attMapBaseFilename, outputPath, [2.08625 2.08625 2.03125], 21, 3)');
        return;
    end
end


%% READING THE SINOGRAMS
disp('Converting input sinogram to APIRL format...');
% Read the sinograms:
[pathstr,name,ext] = fileparts(sinogramFilename);
outFilenameIntfSinograms = [outputPath pathBar 'sinogramSpan1'];
[sinogramSpan1, delaySinogramSpan1, structSizeSino3dSpan1] = getIntfSinogramsFromUncompressedMmr(sinogramFilename, outFilenameIntfSinograms);
% Rewrite the sinogram filename to be used in the next operations:
sinogramFilename = [outFilenameIntfSinograms];
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
   create_norm_files_mmr(normFilename, [], [], [], [], 1);
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
    acfFilename = ['acfsSinogramSpan1'];
    acfsSinogramSpan1 = createACFsFromImage(attenMap, imageSizeAtten_mm, outputPath, acfFilename, sinogramFilename, structSizeSino3dSpan1, 0);

    % After the projection read the acfs:
    acfFilename = [outputPath acfFilename];
    fid = fopen([acfFilename '.i33'], 'r');
    numSinos = sum(structSizeSino3dSpan1.sinogramsPerSegment);
    [acfsSinogramSpan1, count] = fread(fid, structSizeSino3dSpan1.numTheta*structSizeSino3dSpan1.numR*numSinos, 'single=>single');
    acfsSinogramSpan1 = reshape(acfsSinogramSpan1, [structSizeSino3dSpan1.numR structSizeSino3dSpan1.numTheta numSinos]);
    % Close the file:
    fclose(fid);
else
    acfFilename = '';
end
%% GENERATE AND SAVE ATTENUATION AND NORMALIZATION FACTORS AND CORECCTION FACTORS FOR SPAN11 SINOGRAMS FOR APIRL
disp('Generating the ANF sinogram...');
% Save:
outputSinogramName = [outputPath '/NF_Span1'];
interfileWriteSino(single(overall_nf_3d), outputSinogramName, structSizeSino3dSpan1);

% Compose with acfs:
atteNormFactorsSpan1 = overall_nf_3d;
atteNormFactorsSpan1(acfsSinogramSpan1 ~= 0) = overall_nf_3d(acfsSinogramSpan1 ~= 0) ./acfsSinogramSpan1(acfsSinogramSpan1 ~= 0);
anfFilename = [outputPath '/ANF_Span1'];
interfileWriteSino(single(atteNormFactorsSpan1), anfFilename, structSizeSino3dSpan1);
clear atteNormFactorsSpan1;
%clear normFactorsSpan11;
%% GENERATE OSEM AND MLEM RECONSTRUCTION FILES FOR APIRL
disp('Osem reconstruction...');
saveInterval = 1;
saveIntermediate = 0;
outputFilenamePrefix = [outputPath 'reconImage'];
filenameOsemConfig = [outputPath sprintf('osem_%dsubsets.par', numSubsets)];
CreateOsemConfigFileForMmr(filenameOsemConfig, [sinogramFilename '.h33'], [filenameInitialEstimate '.h33'], outputFilenamePrefix, numIterations, [],...
    numSubsets, saveInterval, saveIntermediate, [], [], [], [anfFilename '.h33']);

%% RECONSTRUCTION OF HIGH RES IMAGE
% Execute APIRL: 
status = system(['OSEM ' filenameOsemConfig]) 
%% READ RESULTS
% Read interfile reconstructed image:
volume = interfileRead([outputFilenamePrefix '_final.h33']);