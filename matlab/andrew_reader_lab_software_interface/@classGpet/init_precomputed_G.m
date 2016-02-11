% *********************************************************************
% Reconstruction Framework for Siemens Biograph mMR.  
% class: Gpet
% Authors: Martin Belzunce, Abolfazl Mehranian. Kings College London.
% Date: 08/02/2016
% *********************************************************************
function g = init_precomputed_G (objGpet)
    if isempty (objGpet.tempPath) || (~isempty (objGpet.tempPath)&& ~exist([objGpet.tempPath '\gantry_model_info.mat'],'file'))
        opt.span       = objGpet.sinogram_size.span;
        opt.nseg       = objGpet.sinogram_size.nSeg;
        opt.iRows      = objGpet.image_size.matrixSize(1);
        opt.iColumns   = objGpet.image_size.matrixSize(2);
        opt.iSlices    = objGpet.image_size.matrixSize(3);
        opt.iVoxelSize = objGpet.image_size.voxelSize_mm* 0.1;%cm
        opt.Dir        = objGpet.tempPath;
        fprintf('System matrix pre-computation:...\n')
        fprintf('Span: %d, nSeg: %d, imMatrix: %d x %d x %d\n', opt.span,opt.nseg,opt.iRows,opt.iColumns, opt.iSlices )
        g = Biograph_mMR_S02(opt);

        if isempty (objGpet.tempPath)
         objGpet.tempPath = g.Dir;
        end
    else
        g = load ([objGpet.tempPath '\gantry_model_info.mat']); g = g.g;
    end

    if isempty (objGpet.Geom)
        fprintf('Load pre-computed parts of system matrix into Geom...');
        objGpet.Geom  = cell(1,length(g.AxialSymPlanes));
        for i = 1:length(g.AxialSymPlanes)
            P = load([g.Dir '\Matrix_' num2str(i) '.mat']); P = P.sMatrix;
            objGpet.Geom {i} = P;
            clear P
        end
        fprintf('Done\n');
    end
end