function [SumGrad] = gpuGradient(ObjPrior,Img)
    kernel_gradient = parallel.gpu.CUDAKernel('gradient.ptx', 'gradient.cu');
                                 
end