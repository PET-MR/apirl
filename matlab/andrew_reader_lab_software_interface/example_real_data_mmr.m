%% EXAMPLE MLEM MARTIN PROJECTOR (ANY SPAN)
clear all 
close all
%% CONFIGURE PATHS
% Check what OS I am running on:
if(strcmp(computer(), 'GLNXA64'))
    os = 'linux';
    pathBar = '/';
    sepEnvironment = ':';
elseif(strcmp(computer(), 'PCWIN') || strcmp(computer(), 'PCWIN64'))
    os = 'windows';
    pathBar = '\';
    sepEnvironment = ';';
else
    disp('OS not compatible');
    return;
end

% APIRL PATH
apirlPath = '/home/mab15/workspace/apirl-code/trunk/';
addpath(genpath([apirlPath pathBar 'matlab']));
setenv('PATH', [getenv('PATH') sepEnvironment apirlPath pathBar 'build' pathBar 'bin']);
setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') sepEnvironment apirlPath pathBar 'build' pathBar 'bin']);
%% INIT CLASS GPET
PET.scanner = 'mMR';
PET.method =  'otf_siddon_cpu';
PET.PSF.type = 'none';
PET.radialBinTrim = 0;
PET.Geom = '';
PET.method_for_randoms = 'from_ML_singles_matlab'; %'from_e7_binary_interfile';
PET.method_for_scatter = 'from_e7_binary_interfile';
% To change span:
PET.sinogram_size.span = -1; % Any span, 0 for multislice 2d, -1 for 2d.
PET.nSubsets = 1;
PET = classGpet(PET);
%% EMISSION SINOGRAM
[sinogram, delayedSinogram, structSizeSino3d] = interfileReadSino('/media/mab15/DATA_BACKUP/Scans/PatientData/FDG_Patient_01/e7/PETSinoPlusUmap-Converted/PETSinoPlusUmap-00/PETSinoPlusUmap-00-sino-uncomp.s.hdr');
if PET.sinogram_size.span > 1
    sino_compressed = PET.apply_axial_compression_from_span1(sinogram);
elseif PET.sinogram_size.span == 0
    % Get the direct sinograms:
    sino_compressed = sinogram(:,:,1:PET.sinogram_size.nRings);
    delayedSinogram = delayedSinogram(:,:,1:PET.sinogram_size.nRings);
elseif PET.sinogram_size.span == -1
    % Get the central slice:
    ring = 32;
    sino_compressed = sinogram(:,:,ring);
    delayedSinogram = delayedSinogram(:,:,ring);
end
%% NORM
ncfs = PET.NCF(); % time-invariant.
%ncfs = PET.NCF('PETSinoPlusUmap-norm.n'); % time-variant.
%% ACFs
attenuationMap_filename = '/media/mab15/DATA_BACKUP/Scans/PatientData/FDG_Patient_01/e7/PETSinoPlusUmap-Converted/PETSinoPlusUmap-00/PETSinoPlusUmap-00-umap.v.hdr';
attenuationMapHardware_filename = '/media/mab15/DATA_BACKUP/Scans/PatientData/FDG_Patient_01/e7/PETSinoPlusUmap-Converted/PETSinoPlusUmap-00/PETSinoPlusUmap-00-umap-hardware.v.hdr';
[attenuationMap refMuMap] = interfileReadSiemensImage(attenuationMap_filename);
[attenuationMapHardware refMuMap] = interfileReadSiemensImage(attenuationMapHardware_filename);
attenuationMap = attenuationMap + attenuationMapHardware;
if PET.sinogram_size.span == -1
    % Get only
    slice = 2*ring; % atten map has 127 slices and the direct rings 64.
    attenuationMap = attenuationMap(:,:,slice);
end
acfs = PET.ACF(attenuationMap, refMuMap);
%% RANDOMS
% if uses the randomsBinaryFilename = '/media/mab15/DATA/PatientData/FDG/PETSinoPlusUmap-Converted/PETSinoPlusUmap-00/temp/smoothed_rand_00.s';
% randoms = PET.R(randomsBinaryFilename);
randoms = PET.R(delayedSinogram);
%% SCATTER
scatterBinaryFilename = '/media/mab15/DATA_BACKUP/Scans/PatientData/FDG_Patient_01/e7/PETSinoPlusUmap-Converted/PETSinoPlusUmap-00/rawdata_sino/scatter_estim2d_000000.s';
% for 3d or multislice 2d:
if PET.sinogram_size.span >=0
    scatter = PET.S(scatterBinaryFilename, sino_compressed, ncfs, acfs, randoms); % Needs all that parameters to scale it.
else
    % For 2d select the sinogram ring:
    scatter = PET.S(scatterBinaryFilename, sino_compressed, ncfs, acfs, randoms, ring);
end
% This scatter is already normalzied
%% SENSITIVITY IMAGE
anf = acfs .* ncfs;
anf(anf~=0) = 1./anf(anf~=0);
sensImage = PET.Sensitivity(anf);
%% OP-OSEM
% additive term:
additive = (randoms + scatter).*ncfs.*acfs; % (randoms +scatter)./(afs*nfs) = (randoms+scatter)+
additive = zeros(size(additive));
recon = PET.ones();
recon = PET.OPOSEM(sino_compressed,additive, sensImage,recon, ceil(60/PET.nSubsets));