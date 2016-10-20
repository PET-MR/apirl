function set_framework_environment(apirlPath, apirlBinaryPath)

if nargin == 0
    % APIRL PATH
    apirlPath = '../../';
    apirlBinaryPath = apirlPath;
elseif nargin == 1
    apirlBinaryPath = apirlPath;
end

%% CONFIGURE PATHS
% Check what OS I am running on:
if(strcmp(computer(), 'GLNXA64'))
    os = 'linux';
    pathBar = '/';
    sepEnvironment = ':';
    setenv('PATH', [getenv('PATH') sepEnvironment apirlBinaryPath pathBar 'build' pathBar 'bin']);
    setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') sepEnvironment apirlBinaryPath pathBar 'build' pathBar 'bin']);
elseif(strcmp(computer(), 'PCWIN') || strcmp(computer(), 'PCWIN64'))
    os = 'windows';
    pathBar = '\';
    sepEnvironment = ';';
    setenv('PATH', [getenv('PATH') sepEnvironment apirlBinaryPath pathBar 'build' pathBar 'bin']);
    setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') sepEnvironment apirlBinaryPath pathBar 'build' pathBar 'bin']);
else
    disp('OS not compatible');
    return;
end
% Matlab path:
addpath(genpath([apirlPath pathBar 'matlab']));
