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
%%
PETData = PETDataClass(); % prompts a window to locate rawdata file

% user provides the path
PETData = PETDataClass('E:\mMR\BRAIN_PETMR\brain_68');

% prompts a window to select rawdata folder and sets modifies the default parameters
opt.Span= 1;
opt.MethodSinoData= 'e7';
PETData = PETDataClass([],opt);

% user provides the path and modifies default  settings
PETData = PETDataClass('E:\mMR\BRAIN_PETMR\brain_68',opt); 

% new methods
PETData.uncompress(); % file-name or prompts a window to locate it
PETData.PlotMichelogram
PETData.SinogramDisplay
PETData.ACF2 % human-only mu-map
PETData.Scatters; %3D scatters

