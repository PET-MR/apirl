% *********************************************************************
% Reconstruction Framework for Siemens Biograph mMR.  
% class: Gpet
% Authors: Martin Belzunce, Abolfazl Mehranian. Kings College London.
% Date: 26/02/2016
% *********************************************************************
% Computes the scatter using one of the availables methods.
% Ways to call it:
% R(ObjPET, delayedSinogram, ncf);
function r=R(varargin)
    objGpet = varargin{1};
    r = [];
    if ~strcmpi(objGpet.scanner,'mMR')
        error('Randoms are only available for mMR scanner.');
    end
    
    if nargin == 2 % Simple simulation, smooth a sinogram.
        r = zeros(size(varargin{2}));
        for i = 1 : size(s,3)
            s(:,:,i) = imfilter(varargin{2}(:,:,i), h, 'same');
        end
        % Fit tail?
    elseif nargin == 3 % Sss simulation, need activty image and attenuation.
        if strcmpi(objGpet.scatter_algorithm,'e7_tools')
            % Call e7 tools:
        elseif strcmpi(objGpet.scatter_algorithm,'from_ML_singles_stir')
            structSizeSino3d = getSizeSino3dFromSpan(objGpet.sinogram_size.nRadialBins, objGpet.sinogram_size.nAnglesBins, ...
                objGpet.sinogram_size.nRings, 596/2, 260, ...
                objGpet.sinogram_size.span, objGpet.sinogram_size.maxRingDifference);
            % Call stir:
            [r, structSizeSino] = estimateRandomsWithStir(varargin{2}, structSizeSino3d, varargin{3}, structSizeSino3d, outputPath);
        elseif strcmpi(objGpet.scatter_algorithm,'from_ML_singles_stir')
            structSizeSino3d = getSizeSino3dFromSpan(objGpet.sinogram_size.nRadialBins, objGpet.sinogram_size.nAnglesBins, ...
                objGpet.sinogram_size.nRings, 596/2, 260, ...
                objGpet.sinogram_size.span, objGpet.sinogram_size.maxRingDifference);
            % Call matlab function in apirl:
            [r, singlesOut] = estimateRandomsFromDelayeds(varargin{2}, structSizeSino3d, numIterations, varargin{3}, structSizeSino3d, outputPath);
        end
    end
 end