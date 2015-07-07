clear all
close all
apirlPath = '/workspaces/Martin/KCL/apirl-code/trunk/';
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
% SET ENVIRONMENT AND MATLAB PATHS
addpath(genpath([apirlPath pathBar 'matlab']));
setenv('PATH', [getenv('PATH') sepEnvironment apirlPath pathBar 'build' pathBar 'bin']);
setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') sepEnvironment apirlPath pathBar 'build' pathBar 'bin']);
%%
pathMrDicom = '/media/martin/My Book/BackupWorkspace/KCL/Biograph_mMr/Mediciones/BRAIN_PETMR/T1_fl2D_TRA/';
%pathMrDicom = '/media/martin/My Book/BRAIN_PETMR/Dixon_1/';
%pathMrDicom = '/media/martin/My Book/BRAIN_PETMR/UTE/';
pathMrDicom = '/media/martin/My Book/BackupWorkspace/KCL/Biograph_mMr/Mediciones/BRAIN_PETMR/PETMR_recons/AC_WDIXON_2/';
baseDicomFilename = '';
%pathDicom = '/media/martin/My Book/BackupWorkspace/KCL/Biograph_mMr/Mediciones/LineSources/Line_Source_recon/linesource/';


% dicomInfo.ImagePositionPatient
% dicomInfo.PixelSpacing
% dicomInfo.SliceLocation
% dicomInfo.ImageOrientationPatient
%% INTERFILE
interfileHeaderPet = '/media/martin/My Book/BackupWorkspace/KCL/Biograph_mMr/Mediciones/BRAIN_PETMR/SINOGRAMS/PET_ACQ_68_20150610155347_ima_AC_000_000.v.hdr';

[mrRescaled, petImage, refMrRescaled] = RemapMrIntoPetImageSpace(interfileHeaderPet, pathMrDicom, baseDicomFilename, 0);
interfilewrite(mrRescaled, 'test');
interfilewrite(petImage, 'test2');