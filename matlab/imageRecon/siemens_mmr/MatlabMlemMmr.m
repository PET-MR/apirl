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
%  The span for the reconstruction is taken from the sinogram.
%
%  The save interval parameter permits to store data of each "saveInterval"
%  iterations, if zero, only temporary files are written in a temp folder
%  and are overwritten for each iteration.
% Examples:
%   volume = MatlabMlemMmrSpan1(sinogramFilename, normFilename, attMapBaseFilename, outputPath, pixelSize_mm, numInputIterations, saveInterval, useGpu)

function volume = MatlabMlemMmr(sinogramFilename, normFilename, attMapBaseFilename, outputPath, pixelSize_mm, numInputIterations, saveInterval, useGpu)

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
%filenameInitialEstimate = [outputPath pathBar 'initialEstimate'];
%interfilewrite(initialEstimate, filenameInitialEstimate, pixelSize_mm);
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
%% MLEM
% Runs ordinary poission mlem.
disp('###################### ML-EM RECONSTRUCTION ##########################');
% 1) Compute sensitivity image. Backprojection of attenuation and
% normalization factors:
disp('Compute sensitivity image...');
anfSino = overall_ncf_3d .* acfsSinogram; % ancf.
anfSino(anfSino~= 0) = 1./anfSino(anfSino~= 0); % anf.
% Backproject sinogram in other path:
sensitivityPath = [outputPath 'SensitivityImage' pathBar];
mkdir(sensitivityPath);
[sensImage, pixelSize_mm] = BackprojectMmr(anfSino, imageSize_pixels, pixelSize_mm, sensitivityPath, structSizeSino3d.span, [],[], useGpu);
% Generate update threshold:
updateThreshold =  min(min(min(sensImage)))+ ( max(max(max(sensImage)))- min(min(min(sensImage)))) * 0.001;

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
    [projectedImage, structSizeSinogram] = ProjectMmr(emRecon, pixelSize_mm, iterationPath, structSizeSino3d.span, [],[], useGpu);
    % 2.c) Multiply by the anf (this can be avoided if it is also taken
    % from the backprojection):
    projectedImage = projectedImage .* anfSino;
    % 2.d) Divide sinogram by projected sinogram:
    ratioSinograms = zeros(size(projectedImage), 'single');
    ratioSinograms(projectedImage~=0) = sinograms(projectedImage~=0) ./ projectedImage(projectedImage~=0);
    ratioSinograms = ratioSinograms.* anfSino; % Apply anf.
    % 2.e) Backprojection:
    [backprojImage, pixelSize_mm] = BackprojectMmr(ratioSinograms, imageSize_pixels, pixelSize_mm, iterationPath, structSizeSino3d.span, [],[], useGpu);
    % 2.f) Apply sensitiivty image and correct current image:
    emRecon(sensImage > updateThreshold) = emRecon(sensImage > updateThreshold) .* backprojImage(sensImage > updateThreshold)./ sensImage(sensImage > updateThreshold);
    emRecon(sensImage <= updateThreshold) = 0;
    if rem(iter,saveInterval) == 0
        interfilewrite(emRecon, [outputPath pathBar sprintf('emImage_iter%d', iter)], pixelSize_mm);
    end
end
%% OUTPUT PARAMETER
interfilewrite(emRecon, [outputPath pathBar 'emImage_final'], pixelSize_mm);
volume = emRecon;