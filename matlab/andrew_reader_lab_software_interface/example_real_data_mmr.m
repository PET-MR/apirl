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
apirlPath = 'E:\apirl-code\trunk\';
addpath(genpath([apirlPath pathBar 'matlab']));
setenv('PATH', [getenv('PATH') sepEnvironment apirlPath pathBar 'build' pathBar 'bin']);
setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') sepEnvironment apirlPath pathBar 'build' pathBar 'bin']);
%% INIT CLASS GPET
PET.scanner = 'mMR';
PET.method =  'otf_siddon_gpu';
PET.PSF.type = 'none';
PET.radialBinTrim = 0;
PET.Geom = '';
PET.method_for_randoms = 'from_ML_singles_matlab'; %'from_e7_binary_interfile';
PET.method_for_scatter = 'from_e7_binary_interfile';
PET = classGpet(PET);
%% EMISSION SINOGRAM
[sinogram, delayedSinogram, structSizeSino3d] = interfileReadSino('/media/mab15/DATA/PatientData/FDG/PETSinoPlusUmap-Converted/PETSinoPlusUmap-00/PETSinoPlusUmap-00-sino-uncomp.s.hdr');
sino_compressed = PET.apply_axial_compression_from_span1(sinogram);
%% NORM
ncfs = PET.NCF(); % time-invariant.
%ncfs = PET.NCF('PETSinoPlusUmap-norm.n'); % time-variant.
%% ACFs
attenuationMap_filename = '/media/mab15/DATA/PatientData/FDG/PETSinoPlusUmap-Converted/PETSinoPlusUmap-00/PETSinoPlusUmap-00-umap.v.hdr';
attenuationMapHardware_filename = '/media/mab15/DATA/PatientData/FDG/PETSinoPlusUmap-Converted/PETSinoPlusUmap-00/PETSinoPlusUmap-00-umap-hardware.v.hdr';
[attenuationMap refMuMap] = interfileReadSiemensImage(attenuationMap_filename);
[attenuationMapHardware refMuMap] = interfileReadSiemensImage(attenuationMapHardware_filename);
attenuationMap = attenuationMap + attenuationMapHardware;
acfs = PET.ACF(attenuationMap, refMuMap);
%% RANDOMS
% if uses the randomsBinaryFilename = '/media/mab15/DATA/PatientData/FDG/PETSinoPlusUmap-Converted/PETSinoPlusUmap-00/temp/smoothed_rand_00.s';
% randoms = PET.R(randomsBinaryFilename);
randoms = PET.R(delayedSinogram);
%% SCATTER
scatterBinaryFilename = '/media/mab15/DATA/PatientData/FDG/PETSinoPlusUmap-Converted/PETSinoPlusUmap-00/temp/scatter_estim2d_000000.s';
scatter = PET.S(scatterBinaryFilename, sino_compressed, ncfs, acfs, randoms); % Needs all that parameters to scale it.
% This scatter is already normalzied
%% SENSITIVITY IMAGE
anf = acfs .* ncfs;
anf(anf~=0) = 1./anf(anf~=0);
sensImage = PET.Sensitivity(anf);
%% OP-OSEM
% additive term:
additive = (randoms + scatter).*ncfs.*acfs; % (randoms +scatter)./(afs*nfs) = (randoms+scatter)
recon = PET.ones();
recon = PET.OPOSEM(sino_compressed,additive, sensImage,recon, ceil(60/PET.nSubsets));