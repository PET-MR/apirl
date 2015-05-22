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
%  It receives two optional parameters
%  - pixelSize_mm: that is a vector with 3 elements [pixelSizeX_mm
%  pixelSizeY_mm pixelSizeZ_mm].
%  - numIterations: number of iterations.
%
%  To run the fastest version, use MlemMmrSpan1 that runs the binary APIRL
%  version directly. This version runs with APIRL only the projection and
%  backprojection. Stores each iteration in a different folder from the bas
%  outputPath.
%  
%  The save interval parameter permits to store data of each "saveInterval"
%  iterations, if zero, only temporary files are written in a temp folder
%  and are overwritten for each iteration.
% Examples:
%   volume = MatlabMlemMmrSpan1(sinogramFilename, normFilename, attMapBaseFilename, outputPath, pixelSize_mm, numInputIterations, saveInterval, useGpu)

function volume = MatlabMlemMmrSpan1(sinogramFilename, normFilename, attMapBaseFilename, outputPath, pixelSize_mm, numInputIterations, saveInterval, useGpu)

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
numIterations = 40;
if nargin < 5
    % Default pixel size:
    pixelSize_mm = [2.08625 2.08625 2.03125];
    imageSize_pixels = [286 286 127]; 
    imageSize_mm = pixelSize_mm.*imageSize_pixels;
    useGpu = 0;
    saveInterval = 0;
else
    imageSize_mm = [600 600 257];
    imageSize_pixels = ceil(imageSize_mm./pixelSize_mm);
    if nargin == 8
        numIterations = numInputIterations;
    else
        error('Wrong number of parameters: volume = OsemMmrSpan1(sinogramFilename, normFilename, attMapBaseFilename, outputPath, [2.08625 2.08625 2.03125], 60)');
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
%filenameInitialEstimate = [outputPath pathBar 'initialEstimate'];
%interfilewrite(initialEstimate, filenameInitialEstimate, pixelSize_mm);
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
    acfsSinogramSpan1 = createACFsFromImage(attenMap, imageSizeAtten_mm, outputPath, acfFilename, sinogramFilename, structSizeSino3dSpan1, 0, useGpu);

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
%% MLEM
% Runs ordinary poission mlem.
disp('###################### ML-EM RECONSTRUCTION ##########################');
% 1) Compute sensitivity image. Backprojection of attenuation and
% normalization factors:
disp('Compute sensitivity image...');
anfSino = overall_ncf_3d .* acfsSinogramSpan1; % ancf.
anfSino(anfSino~= 0) = 1./anfSino(anfSino~= 0); % anf.
% Backproject sinogram in other path:
sensitivityPath = [outputPath pathBar 'SensitivityImage/'];
mkdir(sensitivityPath);
[sensImage, pixelSize_mm] = BackprojectMmrSpan1(anfSino, imageSize_pixels, pixelSize_mm, sensitivityPath, useGpu);

% 2) Reconstruction.
emRecon = initialEstimate; % em_recon is the current reconstructed image.

for iter = 1 : numIterations
    disp(sprintf('Iteration %d...', iter));
    % 2.a) Create working directory:
    if rem(iter,saveInterval) == 0
        iterationPath = [outputPath pathBar sprintf('Iteration%d', iter) pathBar];
        mkdir(iterationPath);
    else
        iterationPath = [outputPath 'temp' pathBar];
        mkdir(iterationPath);
    end
    % 2.b) Project current image:
    [projectedImage, structSizeSinogram] = ProjectMmrSpan1(emRecon, pixelSize_mm, iterationPath, useGpu);
    % 2.c) Multiply by the anf (this can be avoided if it is also taken
    % from the backprojection):
    projectedImage = projectedImage .* anfSino;
    % 2.d) Divide sinogram by projected sinogram:
    ratioSinograms = zeros(size(projectedImage), 'single');
    ratioSinograms(projectedImage~=0) = sinogramSpan1(projectedImage~=0) ./ projectedImage(projectedImage~=0);
    % 2.e) Backprojection:
    [backprojImage, pixelSize_mm] = BackprojectMmrSpan1(ratioSinograms, imageSize_pixels, pixelSize_mm, iterationPath, useGpu);
    % 2.f) Apply sensitiivty image and correct current image:
    emRecon(sensImage~=0) = emRecon(sensImage~=0) .* backprojImage(sensImage~=0)./ sensImage(sensImage~=0);
    emRecon(sensImage==0) = 0;
    if rem(iter,saveInterval) == 0
        interfilewrite(emRecon, [outputPath pathBar sprintf('emImage_iter%d', iter)], pixelSize_mm);
    end
end
%% OUTPUT PARAMETER
interfilewrite(emRecon, [outputPath pathBar 'emImage_final'], pixelSize_mm);
volume = emRecon;