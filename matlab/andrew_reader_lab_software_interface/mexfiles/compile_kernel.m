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
% Copy into the priors class to be used:
%copyfile('mexGPUGradient.mexa64', mexInstallPath);