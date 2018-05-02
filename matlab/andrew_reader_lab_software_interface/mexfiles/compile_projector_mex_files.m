% script that compiles kernels
% mexInstallPath = '../../mexfiles/';
% if ~isdir(mexInstallPath)
%     mkdir(mexInstallPath)
% end

%mexcuda gradient_kernel.cu -lstdc++ -lc
mex mexProject.cpp -lstdc++ -lc -lrecon -lreconGPU -ldata -L../../../build/bin/ -I../../../data/inc -I../../../recon/inc -I../../../reconGPU/inc -I../../../recon/inc -I../../../utils/inc -I/usr/local/cuda/include

mex mexBackproject.cpp -lstdc++ -lc -lrecon -lreconGPU -ldata -L../../../build/bin/ -I../../../data/inc -I../../../recon/inc -I../../../reconGPU/inc -I../../../recon/inc -I../../../utils/inc -I/usr/local/cuda/include

