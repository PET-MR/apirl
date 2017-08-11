% script that compiles kernels
% mexInstallPath = '../../mexfiles/';
% if ~isdir(mexInstallPath)
%     mkdir(mexInstallPath)
% end

[result, message] = system('nvcc -ptx -arch=sm_35 kernel_gradient.cu -lstdc++ -lc');
if result == 0
    disp('Compilation of cuda kernel succesfull.');
else
    error(['Error in compilation of cuda kernels: ' message]);
end
%mexcuda gradient_kernel.cu -lstdc++ -lc

mexcuda mexGPUGradient.cu -lstdc++ -lc
mexcuda mexGPUGradientWithSimilarityKernel.cu -lstdc++ -lc
% Another way of compiling:
%system('nvcc -c -arch=sm_35 mexGPUGradientWithSimilarityKernel.cu -I/usr/local/MATLAB/R2017b/extern/include -I/usr/local/MATLAB/R2017b/toolbox/distcomp/gpu/extern/include -lstdc++ -lc')
%mex -g -largeArrayDims mexGPUGradientWithSimilarityKernel.o -L/usr/local/cuda/lib64 -L/usr/local/MATLAB/R2017b/bin/glnxa64 -lc -lstdc++
% Copy into the priors class to be used:
%copyfile('mexGPUGradient.mexa64', mexInstallPath);