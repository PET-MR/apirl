clear all
close all

apirlPath = [fileparts(mfilename('fullpath')) filesep '..' filesep '..' filesep '..'];
% SET ENVIRONMENT AND MATLAB PATHS
addpath(genpath([apirlPath filesep 'matlab']));
setenv('PATH', [getenv('PATH') pathsep apirlPath filesep 'build' filesep 'bin']);
setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') pathsep apirlPath filesep 'build' filesep 'bin']);
%% OUTPUT PATH
% Save all the resampled images in interfile format in an output path:
outputPath = '/workspaces/Martin/KCL/mixed_pet_mr_reconstruction/TestRemapImages/';
if ~isdir(outputPath)
    mkdir(outputPath)
end
visualization = 0;
%% INTERFILE PET IMAGE
interfileHeaderPet = '/workspaces/Martin/KCL/Biograph_mMr/Mediciones/BRAIN_PETMR/SINOGRAMS/PET_ACQ_68_20150610155347_ima_AC_000_000.v.hdr';
%% MR IMAGES
pathMrDicom = '/media/martin/My Book/BackupWorkspace/KCL/Biograph_mMr/Mediciones/BRAIN_PETMR/T1_fl2D_TRA/';
baseDicomFilename = '';
[mrRescaled, petImage, refMrRescaled] = RemapMrIntoPetImageSpace(interfileHeaderPet, pathMrDicom, baseDicomFilename, visualization);
interfilewrite(petImage, [outputPath 'pet'], [refMrRescaled.PixelExtentInWorldY refMrRescaled.PixelExtentInWorldX refMrRescaled.PixelExtentInWorldZ]);
interfilewrite(mrRescaled, [outputPath 'mr_1'], [refMrRescaled.PixelExtentInWorldY refMrRescaled.PixelExtentInWorldX refMrRescaled.PixelExtentInWorldZ]);
%% MR IMAGES
pathMrDicom = '/media/martin/My Book/BackupWorkspace/KCL/Biograph_mMr/Mediciones/BRAIN_PETMR/Dixon_1/';
baseDicomFilename = {'BRAIN_PETMR.MR.CDT_PLANT_SINGLE_BED_PETMR.0003','BRAIN_PETMR.MR.CDT_PLANT_SINGLE_BED_PETMR.0004',...
    'BRAIN_PETMR.MR.CDT_PLANT_SINGLE_BED_PETMR.0005', 'BRAIN_PETMR.MR.CDT_PLANT_SINGLE_BED_PETMR.0006'};
for i = 1 : numel(baseDicomFilename)
    [mrRescaled, petImage, refMrRescaled] = RemapMrIntoPetImageSpace(interfileHeaderPet, pathMrDicom, baseDicomFilename{i}, visualization);
    interfilewrite(mrRescaled, [outputPath sprintf('mr_2_%d',i)], [refMrRescaled.PixelExtentInWorldY refMrRescaled.PixelExtentInWorldX refMrRescaled.PixelExtentInWorldZ]);
end
%% MR IMAGES
pathMrDicom = '/media/martin/My Book/BackupWorkspace/KCL/Biograph_mMr/Mediciones/BRAIN_PETMR/UTE/';
baseDicomFilename = '';
[mrRescaled, petImage, refMrRescaled] = RemapMrIntoPetImageSpace(interfileHeaderPet, pathMrDicom, baseDicomFilename, visualization);
interfilewrite(mrRescaled, [outputPath 'mr_3'], [refMrRescaled.PixelExtentInWorldY refMrRescaled.PixelExtentInWorldX refMrRescaled.PixelExtentInWorldZ]);
%% MR IMAGE
pathMrDicom = '/media/martin/My Book/BackupWorkspace/KCL/Biograph_mMr/Mediciones/BRAIN_PETMR/Head_T1_fl2D_sag_2/';
baseDicomFilename = '';
[mrRescaled, petImage, refMrRescaled] = RemapMrIntoPetImageSpace(interfileHeaderPet, pathMrDicom, baseDicomFilename, visualization);
interfilewrite(mrRescaled, [outputPath 'mr_4'], [refMrRescaled.PixelExtentInWorldY refMrRescaled.PixelExtentInWorldX refMrRescaled.PixelExtentInWorldZ]);
%% MR IMAGE
pathMrDicom = '/media/martin/My Book/BackupWorkspace/KCL/Biograph_mMr/Mediciones/BRAIN_PETMR/ep2d_diff_3scan_trace_p2/';
baseDicomFilename = '';
[mrRescaled, petImage, refMrRescaled] = RemapMrIntoPetImageSpace(interfileHeaderPet, pathMrDicom, baseDicomFilename, visualization);
interfilewrite(mrRescaled, [outputPath 'mr_5'], [refMrRescaled.PixelExtentInWorldY refMrRescaled.PixelExtentInWorldX refMrRescaled.PixelExtentInWorldZ]);
%% MR IMAGE
pathMrDicom = '/media/martin/My Book/BackupWorkspace/KCL/Biograph_mMr/Mediciones/BRAIN_PETMR/ep2d_diff_3scan_trace_p2_ADC/';
baseDicomFilename = '';
[mrRescaled, petImage, refMrRescaled] = RemapMrIntoPetImageSpace(interfileHeaderPet, pathMrDicom, baseDicomFilename, visualization);
interfilewrite(mrRescaled, [outputPath 'mr_6'], [refMrRescaled.PixelExtentInWorldY refMrRescaled.PixelExtentInWorldX refMrRescaled.PixelExtentInWorldZ]);
%% MR IMAGE
pathMrDicom = '/media/martin/My Book/BackupWorkspace/KCL/Biograph_mMr/Mediciones/BRAIN_PETMR/30minute_DIXON/';
baseDicomFilename = {'BRAIN_PETMR.MR.CDT_PLANT_SINGLE_BED_PETMR.0012.','BRAIN_PETMR.MR.CDT_PLANT_SINGLE_BED_PETMR.0013.',...
    'BRAIN_PETMR.MR.CDT_PLANT_SINGLE_BED_PETMR.0014.', 'BRAIN_PETMR.MR.CDT_PLANT_SINGLE_BED_PETMR.0015.'};
for i = 1 : numel(baseDicomFilename)
    [mrRescaled, petImage, refMrRescaled] = RemapMrIntoPetImageSpace(interfileHeaderPet, pathMrDicom, baseDicomFilename{i}, visualization);
    interfilewrite(mrRescaled, [outputPath sprintf('mr_%d',6+i)], [refMrRescaled.PixelExtentInWorldY refMrRescaled.PixelExtentInWorldX refMrRescaled.PixelExtentInWorldZ]);
end
%% PET DICOM IMAGE
pathMrDicom = '/workspaces/Martin/KCL/Biograph_mMr/Mediciones/BRAIN_PETMR/PETMR_recons/AC_wDIXON_1/';
baseDicomFilename = '';
[mrRescaled, petImage, refMrRescaled] = RemapMrIntoPetImageSpace(interfileHeaderPet, pathMrDicom, baseDicomFilename, visualization);
interfilewrite(mrRescaled, [outputPath 'pet_dicom_1'], [refMrRescaled.PixelExtentInWorldY refMrRescaled.PixelExtentInWorldX refMrRescaled.PixelExtentInWorldZ]);

