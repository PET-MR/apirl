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
apirlPath = '/workspaces/Martin/apirl-code/trunk/';
addpath(genpath([apirlPath pathBar 'matlab']));
setenv('PATH', [getenv('PATH') sepEnvironment apirlPath pathBar 'build' pathBar 'bin']);
setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') sepEnvironment apirlPath pathBar 'build' pathBar 'bin']);
%% WITH INTERFACE
PETData = PETDataClass(); % prompts a window to locate rawdata file
%% TEST LIST-MODE
% prompts a window to select rawdata folder and sets modifies the default parameters
opt.span= 1;
opt.MethodSinoData= 'e7';
% user provides the path
PETData = PETDataClass('/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/BRAIN_PETMR/listmode', opt);
timeFrame_sec = 5;
PETData.InitFramesConfig(timeFrame_sec);
%%
PETData.ListModeChopper()
%% TEST INTERFILE SINOGRAM
% user provides the path and modifies default  settings
PETData = PETDataClass('/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/BRAIN_PETMR/SINOGRAMS',opt); 

% new methods
PETData.uncompress(); % file-name or prompts a window to locate it
PETData.PlotMichelogram
PETData.SinogramDisplay
PETData.ACF2 % human-only mu-map
PETData.Scatters; %3D scatters

