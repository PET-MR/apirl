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
% CUDA PATH
cudaPath = '/usr/local/cuda/';
setenv('PATH', [getenv('PATH') sepEnvironment cudaPath pathBar 'bin']);
setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') sepEnvironment cudaPath pathBar 'lib64']);
% APIRL PATH
apirlPath = 'F:\workspace\apirl-code\trunk\';
addpath(genpath([apirlPath pathBar 'matlab']));
setenv('PATH', [getenv('PATH') sepEnvironment apirlPath pathBar 'build' pathBar 'bin']);
setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') sepEnvironment apirlPath pathBar 'build' pathBar 'bin']);
%% WITH INTERFACE
%PETData = PETDataClass('/media/mab15/DATA/PatientData/FDG/Raw_PET/'); % prompts a window to locate rawdata file
%% TEST LIST-MODE
% prompts a window to select rawdata folder and sets modifies the default parameters
opt.span= 1;
opt.MethodSinoData= 'e7';
%opt.FrameTimePoints = [0 240];
% user provides the path
PETData = PETDataClass('F:\NEMA_LONG\Scan2');
ncf = PETData.NCF();
PETData.uncompress(PETData.Data.emission.n);
PETData.Reconstruct(21, 3, 0);
PETData.Reconstruct(21, 3, 1);
%PETData = PETDataClass('/media/mab15/DATA/PatientData/Florbetaben/PETListPlusUmap-Converted/PETListPlusUmap-LM-00/', opt);
%timeFrame_sec = 200;
%PETData.InitFramesConfig(timeFrame_sec);
%%
%PETData.ListModeChopper()
% %% TEST INTERFILE SINOGRAM
% % user provides the path and modifies default  settings
% PETData = PETDataClass('/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/BRAIN_PETMR/SINOGRAMS',opt); 
% 
% % new methods
% PETData.uncompress(); % file-name or prompts a window to locate it
% PETData.PlotMichelogram
% PETData.SinogramDisplay
% PETData.ACF2 % human-only mu-map
% PETData.Scatters; %3D scatters
% %%
% Data = PETDataClass('/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/BRAIN_PETMR/SINOGRAMS/');
% 
% AN = Data.muliFactors; % attenuation and normalization factors
% B = Data.addiFactors;  % additive background counts, correctted for attenuation and normalization
% E = Data.Prompts;      % emission data


