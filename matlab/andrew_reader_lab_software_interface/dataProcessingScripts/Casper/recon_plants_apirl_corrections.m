%% EXAMPLE MLEM MARTIN PROJECTOR (ANY SPAN)
clear all 
close all
apirlPath = '/home/mab15/workspace/apirl-code/trunk/';
addpath([apirlPath 'matlab/andrew_reader_lab_software_interface/']);
set_framework_environment(apirlPath);
% set_framework_environment(basePath, binaryPath);
%% DATA PATHS
dataPath = '/data/Scans/Phantoms/Plants2017/raw-plant/';
reconPath = [dataPath 'acq_16/recon_framework_e7_corrections/'];
if ~isdir(reconPath)
    mkdir(reconPath)
end
%sinogramFilename = [dataPath '/data-LM-00-sino-100-0.s.hdr'];
%attenuationMap_filename = [dataPath '/data-LM-00-umap.v.hdr']; % This might be the one overwritten using mumap_registered_2 of reconstruct_without_scatter_and_register_ct
%attenuationMapHardware_filename = [dataPath '/data-LM-00-umap-hardware.v.hdr'];
sinogramFilename = [dataPath '/PET_ACQ_16_20170302131400-0_PRR_1000009_20170315134053-uncomp_00.s.hdr'];
attenuationMap_filename = [dataPath '/PET_ACQ_16_20170302131400-0_PRR_1000009_20170315134053_umap_human_00.v.hdr']; % This might be the one overwritten using mumap_registered_2 of reconstruct_without_scatter_and_register_ct
attenuationMapHardware_filename = [dataPath '/PET_ACQ_16_20170302131400-0_PRR_1000009_20170315134053_umap_hardware_00.v.hdr'];
% Not generated
% normFilename = [dataPath '../data-norm.n'];
% scatterBinaryFilename = [dataPath '/sino_rawdata_100/scatter_estim2d_000000.s'];
% normBinaryFilename = [dataPath '/sino_rawdata_100/norm3d_00.a'];
% randomsBinaryFilename = [dataPath '/sino_rawdata_100/smoothed_rand_00.s'];
% T1_folder = '/data/Scans/PatientData/AD_patients/P01/T1/';
%% PROCESS THE DIFFERENT NOISE LEVELS
%% INIT CLASS GPET
paramPET.scanner = 'mMR';
paramPET.method =  'otf_siddon_gpu';
paramPET.PSF.type = 'none';
paramPET.radialBinTrim = 0;
paramPET.Geom = '';
% paramPET.method_for_normalization = 'from_e7_binary_interfile';
paramPET.method_for_randoms = 'from_ML_singles_matlab'; %
paramPET.method_for_scatter = 'from_e7_binary_interfile';
% To change span:
paramPET.sinogram_size.span = 11; % Any span, 0 for multislice 2d, -1 for 2d.
paramPET.nSubsets = 1;
PET = classGpet(paramPET);

%% READ AND PREPARE DATA
% EMISSION SINOGRAM
[sinogram, delayedSinogram, structSizeSino3d, info] = interfileReadSino(sinogramFilename);
if PET.sinogram_size.span > 1
    sino_span = PET.apply_axial_compression_from_span1(sinogram);
end
% ATTENUATION MAP
[attenuationMapHuman refMuMap] = interfileReadSiemensImage(attenuationMap_filename);
[attenuationMapHardware refMuMap] = interfileReadSiemensImage(attenuationMapHardware_filename);
attenuationMap = attenuationMapHuman + attenuationMapHardware;

PET.setBedPosition(attenuationMap_filename);
% [MrInPet, refMr] = PET.getMrInPetImageSpace(T1_folder);
%% NORM
%ncfs = PET.NCF(normFilename, info.SinglesPerBucket); %
ncfs = PET.NCF();
nf = ncfs;
nf(nf~=0) = 1./nf(nf~=0);
%% RANDOMS
% if uses the randomsBinaryFilename = '/media/mab15/DATA/PatientData/FDG/PETSinoPlusUmap-Converted/PETSinoPlusUmap-00/temp/smoothed_rand_00.s';
%randoms = PET.R(randomsBinaryFilename);
randoms = PET.R(delayedSinogram);
%% ACFs
acfs = PET.ACF(attenuationMap, refMuMap);
acfs_human = PET.ACF(attenuationMapHuman, refMuMap);
%% SCATTER
% for 3d or multislice 2d:
% if PET.sinogram_size.span >=0
%     scatter = PET.S(scatterBinaryFilename, sino_span, ncfs, acfs_human, randoms); % Needs all that parameters to scale it.
% else
%     % For 2d select the sinogram ring:
%     scatter = PET.S(scatterBinaryFilename, sino_span, ncfs, acfs_human, randoms, ring);
% end
scatter = zeros(size(sino_span));

%% OP-MLEM
% SENSITIVITY IMAGE
anf = acfs .* ncfs;
anf(anf~=0) = 1./anf(anf~=0);
sensImage = PET.Sensitivity(anf);
outputPath = [reconPath 'OPMLEM/'];
numIterations = 100;
saveInterval = 10;
% additive term:
additive = randoms + scatter.*nf; % this is the correct way to do it, but AN needs to be included in the projector and backprojector.
initial_image = PET.ones();
opmlem = PET.OPMLEMsaveIter(sino_span, anf, additive, sensImage, initial_image, numIterations, outputPath, saveInterval);
%% OP-MLEM NO ATTEN
% SENSITIVITY IMAGE
anf = ncfs;
anf(anf~=0) = 1./anf(anf~=0);
sensImage = PET.Sensitivity(anf);
outputPath = [reconPath 'OPMLEMnoAtten/'];
numIterations = 100;
saveInterval = 10;
% additive term:
additive = randoms + scatter.*nf; % this is the correct way to do it, but AN needs to be included in the projector and backprojector.
initial_image = PET.ones();
opmlem_no_atten = PET.OPMLEMsaveIter(sino_span, anf, additive, sensImage, initial_image, numIterations, outputPath, saveInterval);
