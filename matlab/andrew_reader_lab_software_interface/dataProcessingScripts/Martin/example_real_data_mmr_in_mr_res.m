%% EXAMPLE MLEM MARTIN PROJECTOR (ANY SPAN)
clear all 
close all
apirlPath = '/home/mab15/workspace/apirl-code/trunk/';
addpath([apirlPath 'matlab/andrew_reader_lab_software_interface/']);
set_framework_environment(apirlPath);
% set_framework_environment(basePath, binaryPath);
%% DATA PATHS
sinogramFilename = '/media/mab15/DATA_BACKUP/Scans/PatientData/FDG_Patient_01/e7/PETSinoPlusUmap-Converted/PETSinoPlusUmap-00/PETSinoPlusUmap-00-sino-uncomp.s.hdr';
attenuationMap_filename = '/media/mab15/DATA_BACKUP/Scans/PatientData/FDG_Patient_01/e7/PETSinoPlusUmap-Converted/PETSinoPlusUmap-00/PETSinoPlusUmap-00-umap.v.hdr';
attenuationMapHardware_filename = '/media/mab15/DATA_BACKUP/Scans/PatientData/FDG_Patient_01/e7/PETSinoPlusUmap-Converted/PETSinoPlusUmap-00/PETSinoPlusUmap-00-umap-hardware.v.hdr';
scatterBinaryFilename = '/media/mab15/DATA_BACKUP/Scans/PatientData/FDG_Patient_01/e7/PETSinoPlusUmap-Converted/PETSinoPlusUmap-00/rawdata_sino/scatter_estim2d_000000.s';
normalizationBinaryFilename = '/media/mab15/DATA_BACKUP/Scans/PatientData/FDG_Patient_01/e7/PETSinoPlusUmap-Converted/PETSinoPlusUmap-00/rawdata_sino/norm3d_00.a';
randomsBinaryFilename = '/media/mab15/DATA_BACKUP/Scans/PatientData/FDG_Patient_01/e7/PETSinoPlusUmap-Converted/PETSinoPlusUmap-00/rawdata_sino/smoothed_rand_00.s';
t1DicomPath = '/media/mab15/DATA_BACKUP/Scans/PatientData/FDG_Patient_01/T1/';
%% INIT CLASS GPET
span = 11;
PET.scanner = 'mMR';
PET.method =  'otf_siddon_gpu';
PET.PSF.type = 'none';
PET.radialBinTrim = 0;
PET.nRays = 1;
PET.nAxialRays = 1;
PET.Geom = '';
PET.method_for_normalization = 'from_e7_binary_interfile';
PET.method_for_randoms = 'from_e7_binary_interfile'; %'from_ML_singles_matlab'; %'from_e7_binary_interfile';
PET.method_for_scatter = 'from_e7_binary_interfile';
% To change span:
PET.sinogram_size.span = span; % Any span, 0 for multislice 2d, -1 for 2d.
PET.nSubsets = 1;
PET = classGpet(PET);

%% READ AND PREPARE DATA
% EMISSION SINOGRAM
[sinogram, delayedSinogram, structSizeSino3d] = interfileReadSino(sinogramFilename);
if PET.sinogram_size.span > 1
    sino_span = PET.apply_axial_compression_from_span1(sinogram);
elseif PET.sinogram_size.span == 0
    % Get the direct sinograms:
    sino_span = sinogram(:,:,1:PET.sinogram_size.nRings);
    delayedSinogram = delayedSinogram(:,:,1:PET.sinogram_size.nRings);
elseif PET.sinogram_size.span == -1
    % Get the central slice:
    ring = 32;
    sino_span = sinogram(:,:,ring);
    delayedSinogram = delayedSinogram(:,:,ring);
else
    % span 1:
    sino_span = sinogram;
end
% ATTENUATION MAP
[attenuationMapHuman refMuMap] = interfileReadSiemensImage(attenuationMap_filename);
[attenuationMapHardware refMuMap] = interfileReadSiemensImage(attenuationMapHardware_filename);
attenuationMap = attenuationMapHuman + attenuationMapHardware;
if PET.sinogram_size.span == -1
    % Get only
    slice = 2*ring; % atten map has 127 slices and the direct rings 64.
    attenuationMap = attenuationMap(:,:,slice);
    attenuationMapHuman = attenuationMapHuman(:,:,slice);
end

% MR IMAGE
% for the mr image is important to have the correct bed position, that is
% set from any interfile header of the scan, for example the attenuation
% map:
PET.setBedPosition(attenuationMap_filename);
% Read Mr image:
[imageMr, refImageMr, imageMrFullFov, refMrImagePetFov] = PET.getMrInNativeImageSpace(t1DicomPath); % Return two images, the original Mr image (imageMr), and the Mr with its original pixel size but for the full fov of the pet scanner (imageMrFullFov)

%% NORM
ncfs = PET.NCF(normalizationBinaryFilename);
%ncfs = PET.NCF('PETSinoPlusUmap-norm.n'); % time-variant.
nf = ncfs;
nf(nf~=0) = 1./nf(nf~=0);
%% ACFs
acfs = PET.ACF(attenuationMap, refMuMap);
acfs_human = PET.ACF(attenuationMapHuman, refMuMap);
%% RANDOMS
% if uses the randomsBinaryFilename = '/media/mab15/DATA/PatientData/FDG/PETSinoPlusUmap-Converted/PETSinoPlusUmap-00/temp/smoothed_rand_00.s';
randoms = PET.R(randomsBinaryFilename);
%randoms = PET.R(delayedSinogram);
%% SCATTER
% for 3d or multislice 2d:
if PET.sinogram_size.span >=0
    scatter = PET.S(scatterBinaryFilename, sino_span, ncfs, acfs_human, randoms); % Needs all that parameters to scale it.
else
    % For 2d select the sinogram ring:
    scatter = PET.S(scatterBinaryFilename, sino_span, ncfs, acfs_human, randoms, ring);
end
% This scatter is already normalzied
        
%% OP-MLEM-DOWNSAMPLE-IN-SYSTEM-MATRIX
outputPath = ['/media/mab15/DATA_BACKUP/Scans/PatientData/FDG_Patient_01/e7/PETSinoPlusUmap-Converted/PETSinoPlusUmap-00/recon_in_mr_res/'];
if ~isdir(outputPath)
    mkdir(outputPath);
end
numIterations = 100; % ceil(60/PET.nSubsets)
saveInterval = 5;
%anf
anf = acfs .* ncfs;
anf(anf~=0) = 1./anf(anf~=0);
% additive:
additive = randoms + scatter.*nf;
% reconstruct
recon = PET.OPMLEM_DS(sino_span, anf, additive, refMrImagePetFov, numIterations, outputPath, saveInterval); % Needs the ref matrix of the new reconstructed image.

%interfilewrite(recon, sprintf('test_span%d_nRays%d_nAxialRays%d', PET.sinogram_size.span, PET.nRays,PET.nAxialRays), PET.image_size.voxelSize_mm);

