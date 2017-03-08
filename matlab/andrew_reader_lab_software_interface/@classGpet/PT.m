% *********************************************************************
% Reconstruction Framework for Siemens Biograph mMR.  
% class: Gpet
% Authors: Martin Belzunce, Abolfazl Mehranian. Kings College London.
% Date: 08/02/2016
% *********************************************************************
% Method that backprojects an image into a singoram.
function x = PT(objGpet,m, subset_i, localNumSubsets)    % If the fourth parameter is received we override the nSubsets of the object with this parameter.  
    %full/sub-backprojection
    if nargin <3
        angles = 1:objGpet.sinogram_size.nAnglesBins;
        subset_i = [];
        numSubsets = [];
    else
        angles = objGpet.sinogram_size.subsets(:,subset_i);
        if nargin == 3
            numSubsets = objGpet.nSubsets;  % Not use directly objGpet.nSubsets, because it canbe the case where there is a number of susbets configured but we still want to project the shile sinogram.
        elseif nargin == 4
            numSubsets = localNumSubsets;
        end
    end
    % Check the image size:
    if size(m) == 1
        % Uniform image with x value:
        m = ones(objGpet.sinogram_size.matrixSize).*m;
    else
        sizeM = size(m);
        if numel(sizeM) == 2
            sizeM = [size(m) 1];% 2d, I need to add the z size because
            % matrixSize is a 3-elements vector.
        end
        if sizeM ~= objGpet.sinogram_size.matrixSize
            warning('m: the input image has a different size to the matrix_size of the proejctor');
        end
    end
    if strcmpi(objGpet.scanner,'2D_radon')
        if strcmpi(objGpet.method, 'otf_matlab')
            x = iradon(m,angles-1,'none',objGpet.image_size.matrixSize(1));
        else
            error(sprintf('The method %s is not available for the scanner %s.', objGpet.method, objGpet.scanner));
        end
    elseif strcmpi(objGpet.scanner,'mMR')
        if strcmpi(objGpet.method, 'pre-computed_matlab')
            g = init_precomputed_G (objGpet);
            RadialBins = (1 : objGpet.sinogram_size.nRadialBins-objGpet.radialBinTrim)+floor(objGpet.radialBinTrim/2);
            x = Project_preComp(objGpet,m,g,angles,RadialBins,-1);
%         elseif strcmpi(objGpet.method, 'otf_siddon_cpu')
%             [x, pixelSize] = BackprojectMmr(m, objGpet.image_size.matrixSize, objGpet.image_size.voxelSize_mm, objGpet.tempPath, objGpet.sinogram_size.span, numSubsets, subset_i, 0);
%         elseif strcmpi(objGpet.method, 'otf_siddon_gpu')
%             [x, pixelSize] = BackprojectMmr(m, objGpet.image_size.matrixSize, objGpet.image_size.voxelSize_mm, objGpet.tempPath, objGpet.sinogram_size.span, numSubsets, subset_i, 1);
%         end
        elseif strcmpi(objGpet.method, 'otf_siddon_cpu')
            structSizeSino = get_sinogram_size_for_apirl(objGpet);
            [x, pixelSize, output_message] = Backproject(m, objGpet.image_size.matrixSize, objGpet.image_size.voxelSize_mm, objGpet.tempPath, objGpet.scanner, objGpet.scanner_properties, structSizeSino, numSubsets, subset_i, 0, objGpet.nRays, objGpet.nAxialRays);
            if objGpet.verbosity > 1
                disp(output_message);
            end
        elseif strcmpi(objGpet.method, 'otf_siddon_gpu')
            structSizeSino = get_sinogram_size_for_apirl(objGpet);
            [x, pixelSize, output_message] = Backproject(m, objGpet.image_size.matrixSize, objGpet.image_size.voxelSize_mm, objGpet.tempPath, objGpet.scanner, objGpet.scanner_properties, structSizeSino, numSubsets, subset_i, 1,objGpet.nRays, objGpet.nAxialRays);
            if objGpet.verbosity > 1
                disp(output_message);
            end
        end
    elseif strcmpi(objGpet.scanner,'cylindrical')
        if strcmpi(objGpet.method, 'pre-computed_matlab')
            disp('todo: is the precomputed version compatible with any cylindrical geometry?.')
        elseif strcmpi(objGpet.method, 'otf_siddon_cpu')
            structSizeSino = get_sinogram_size_for_apirl(objGpet);
            [x, pixelSize, output_message] = Backproject(m, objGpet.image_size.matrixSize, objGpet.image_size.voxelSize_mm, objGpet.tempPath, objGpet.scanner, objGpet.scanner_properties, structSizeSino, numSubsets, subset_i, 0,objGpet.nRays);
            if objGpet.verbosity > 1
                disp(output_message);
            end
        elseif strcmpi(objGpet.method, 'otf_siddon_gpu')
            structSizeSino = get_sinogram_size_for_apirl(objGpet);
            [x, pixelSize, output_message] = Backproject(m, objGpet.image_size.matrixSize, objGpet.image_size.voxelSize_mm, objGpet.tempPath, objGpet.scanner, objGpet.scanner_properties, structSizeSino, numSubsets, subset_i, 1,objGpet.nRays, objGpet.nAxialRays);
            if objGpet.verbosity > 1
                disp(output_message);
            end
        end
    else
        error('unkown scanner')
    end
    
    % PSF convolution
    if strcmpi(objGpet.PSF.type,'shift-invar')
        x = Gauss3DFilter(objGpet, x, objGpet.PSF.Width);
    elseif strcmpi(objGpet.PSF.type,'shift-var')
        disp('todo: shift-var')
    else
        % 'none'
    end
end