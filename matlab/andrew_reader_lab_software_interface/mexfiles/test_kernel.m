% Test kernels:
kernel = parallel.gpu.CUDAKernel('kernel_gradient.ptx', 'kernel_gradient.cu');
Nx = 100; Ny = 75; Nz = 50; Kx = 7; Ky = 7; Kz = 7;
kernel.ThreadBlockSize = [8 8 8];
kernel.GridSize = ceil([Nx Ny Nz]./kernel.ThreadBlockSize); %BLOCK_SIZE_X-2*KERNEL_RADIUS)
A = single(rand(Nx, Ny, Nz));
B = zeros(size(A), 'single');
tic
dA = gpuArray(A);
dB = gpuArray(B);
[dOA, dOB] = feval(kernel, dA, dB,  Nx, Ny, Nz, Kx, Ky, Kz);
toc
%% WITH MEX FILE
tic
grad = mexGPUGradient(A, Kx, Ky, Kz);
toc
%% COMPARE WITH GRAD
apirlPath = '/home/mab15/workspace/apirl-code/trunk/';
addpath([apirlPath 'matlab/andrew_reader_lab_software_interface/']);
set_framework_environment(apirlPath);
opt.PetOptimizationMethod = 'DePierro';%'OSL'
opt.PetPriorType = 'Quadratic'; %'Bowsher' 'JointBurgEntropy'
opt.PetRegularizationParameter = 1;
opt.PetPreCompWeights = 1;
opt.TVsmoothingParameter = 0.1;
opt.LangeDeltaParameter = 1;
opt.BowsherB = 70;
opt.PriorMrImage =[];
opt.MrSigma = 0.1; % JBE
opt.PetSigma  = 10; %JBE
opt.display = 0;
% check if the object already was initilized:
opt.ImageSize = [Nx Ny Nz];
opt.imCropFactor = [0,0,0];
opt.sWindowSize = 7;
opt.lWindowSize = 1;
Prior = PriorsClass(opt);

W0 = opt.PetPreCompWeights./repmat(sum(opt.PetPreCompWeights,2),[1,Prior.nS]); %for weighted quadratic
tic
imgGrad = Prior.GraphGradCrop(A);
imgGrad = reshape(sum(imgGrad,2), [Nx Ny Nz]);
toc