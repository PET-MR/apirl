% *********************************************************************
% Reconstruction Framework for Siemens Biograph mMR.  
% class: Gpet
% Authors: Martin Belzunce, Abolfazl Mehranian. Kings College London.
% Date: 08/02/2016
% *********************************************************************
% Method that projects an image into a singoram.
function m = P(objGpet, x,subset_i, localNumSubsets)    % If the fourth parameter is received we override the nSubsets of the object with this parameter.
    % full/sub-forward
    if nargin <3
        angles = 1:objGpet.sinogram_size.nAnglesBins;
    else
        angles = objGpet.sinogram_size.subsets(:,subset_i);
    end

    % PSF convolution
    if strcmpi(objGpet.PSF.type,'shift-invar')
        x = Gauss3DFilter(objGpet, x, objGpet.PSF.Width);
    elseif strcmpi(objGpet.PSF.type,'shift-var')
        disp('todo: shift-var')
    else
        % 'none'
    end

    % Check the image size:
    if size(x) == 1
        % Uniform image with x value:
        x = ones(objGpet.image_size.matrixSize).*x;
    else
        sizeX = size(x);
        if numel(sizeX) == 2
            sizeX = [size(x) 1];% 2d, I need to add the z size because
            % matrixSize is a 3-elements vector.
        end
        if sizeX ~= objGpet.image_size.matrixSize
            warning('x: the input image has a different size to the matrix_size of the proejctor');
        end
    end  

    if strcmpi(objGpet.scanner,'2D_radon')
        if strcmpi(objGpet.method, 'otf_matlab')
            m = radon(x,angles-1);
        else
            error(sprintf('The method %s is not available for the scanner %s.', objGpet.method, objGpet.scanner));
        end

    elseif strcmpi(objGpet.scanner,'mMR')
        if strcmpi(objGpet.method, 'pre-computed_matlab')
            g = init_precomputed_G (objGpet);
            RadialBins = (1 : objGpet.sinogram_size.nRadialBins-objGpet.radialBinTrim)+floor(objGpet.radialBinTrim/2);
            m = Project_preComp(objGpet,x, g, angles, RadialBins, 1);
        else
            % Select the subsets:
            if nargin < 3
                subset_i = [];
                numSubsets = [];
            elseif nargin == 3
                numSubsets = objGpet.nSubsets;  % Not use directly objGpet.nSubsets, because it canbe the case where there is a number of susbets configured but we still want to project the shile sinogram.
            elseif nargin == 4
                numSubsets = localNumSubsets;
            end
%             if strcmpi(objGpet.method, 'otf_siddon_cpu')
%                 [m, structSizeSinogram] = ProjectMmr(x, objGpet.image_size.voxelSize_mm, objGpet.tempPath, objGpet.sinogram_size.span, numSubsets, subset_i, 0);
%             elseif strcmpi(objGpet.method, 'otf_siddon_gpu')
%                 [m, structSizeSinogram] = ProjectMmr(x, objGpet.image_size.voxelSize_mm, objGpet.tempPath, objGpet.sinogram_size.span, numSubsets, subset_i, 1);
%             end
            structSizeSino = get_sinogram_size_for_apirl(objGpet);
            if strcmpi(objGpet.method, 'otf_siddon_cpu')
                [m, structSizeSinogram, output_message] = Project(x, objGpet.image_size.voxelSize_mm, objGpet.tempPath, objGpet.scanner, objGpet.scanner_properties, structSizeSino, numSubsets, subset_i, 0, objGpet.nRays, objGpet.nAxialRays);
                if objGpet.verbosity > 0
                    disp(output_message);
                end
            elseif strcmpi(objGpet.method, 'otf_siddon_gpu')
                [m, structSizeSinogram, output_message] = Project(x, objGpet.image_size.voxelSize_mm, objGpet.tempPath, objGpet.scanner, objGpet.scanner_properties, structSizeSino, numSubsets, subset_i, 1, objGpet.nRays, objGpet.nAxialRays);
                if objGpet.verbosity > 0
                    disp(output_message);
                end
            end
        end
        
    elseif strcmpi(objGpet.scanner,'cylindrical')
        if strcmpi(objGpet.method, 'pre-computed_matlab')               
            disp('todo: is the precomputed version compatible with any cylindrical geometry?.')
        else
            % Select the subsets:
            if nargin < 3
                subset_i = [];
                numSubsets = [];
            elseif nargin == 3
                numSubsets = objGpet.nSubsets;  % Not use directly objGpet.nSubsets, because it canbe the case where there is a number of susbets configured but we still want to project the shile sinogram.
            elseif nargin == 4
                numSubsets = localNumSubsets;
            end
            structSizeSino = get_sinogram_size_for_apirl(objGpet);
            if strcmpi(objGpet.method, 'otf_siddon_cpu')
                [m, structSizeSinogram, output_message] = Project(x, objGpet.image_size.voxelSize_mm, objGpet.tempPath, objGpet.scanner, objGpet.scanner_properties, structSizeSino, numSubsets, subset_i, 0, objGpet.nRays, objGpet.nAxialRays);
                if objGpet.verbosity > 0
                    disp(output_message);
                end
            elseif strcmpi(objGpet.method, 'otf_siddon_gpu')
                [m, structSizeSinogram, output_message] = Project(x, objGpet.image_size.voxelSize_mm, objGpet.tempPath, objGpet.scanner, objGpet.scanner_properties, structSizeSino, numSubsets, subset_i, 1, objGpet.nRays, objGpet.nAxialRays);
                if objGpet.verbosity > 0
                    disp(output_message);
                end
            end
        end

    else
        error('unkown scanner')
    end
end