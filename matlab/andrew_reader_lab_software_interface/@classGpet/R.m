% *********************************************************************
% Reconstruction Framework for Siemens Biograph mMR.  
% class: Gpet
% Authors: Martin Belzunce, Abolfazl Mehranian. Kings College London.
% Date: 26/02/2016
% *********************************************************************
% Computes the randoms using one of the availables methods.
% Ways to call it:
% R(ObjPET, delayedSinogram, ncf);
function r=R(varargin)
    objGpet = varargin{1};
    r = [];
    if ~strcmpi(objGpet.scanner,'mMR')&& ~strcmpi(objGpet.scanner,'2D_mMR')
        error('Randoms are only available for mMR scanner.');
    end
    param = varargin{2};
    if strcmpi(objGpet.scanner,'mMR')
        if numel(param) == 1 % Simple simulation, constant background with poisson dsitribution, the input parameter is the mean total number of counts.
            counts = varargin{2};
            r = ones(objGpet.sinogram_size.matrixSize);
            meanValue = counts./numel(r);
            r = r .* meanValue;
            % Generate a poisson distributed with contant mean value:
            r =poissrnd(r);
        elseif size(param) == [344 252 4084]% Sss simulation, need activty image and attenuation.
            if strcmpi(objGpet.method_for_randoms,'e7_tools')
                % Call e7 tools:
            elseif strcmpi(objGpet.method_for_randoms,'from_ML_singles_stir')
                structSizeSino3d = getSizeSino3dFromSpan(objGpet.sinogram_size.nRadialBins, objGpet.sinogram_size.nAnglesBins, ...
                    objGpet.sinogram_size.nRings, 596/2, 260, ...
                    objGpet.sinogram_size.span, objGpet.sinogram_size.maxRingDifference);
                % Call stir:
                [r, structSizeSino] = estimateRandomsWithStir(varargin{2}, structSizeSino3d, varargin{3}, structSizeSino3d, outputPath);
            elseif strcmpi(objGpet.method_for_randoms,'from_ML_singles_matlab')
                structSizeSino3d = getSizeSino3dFromSpan(objGpet.sinogram_size.nRadialBins, objGpet.sinogram_size.nAnglesBins, ...
                    objGpet.sinogram_size.nRings, 596/2, 260, ...
                    1, objGpet.sinogram_size.maxRingDifference); % The delayeds are span 1.
                % Call matlab function in apirl:
                numIterations = 3;
                [r, singlesOut] = estimateRandomsFromDelayeds(varargin{2}, structSizeSino3d, numIterations, objGpet.sinogram_size.span);
            end
        end
    elseif strcmpi(objGpet.scanner,'2D_mMR')
        if numel(param) == 1 % Simple simulation, constant background with poisson dsitribution, the input parameter is the mean total number of counts.
            counts = varargin{2};
            r = ones(objGpet.sinogram_size.matrixSize);
            meanValue = counts./numel(r);
            r = r .* meanValue;
            % Generate a poisson distributed with contant mean value:
            r =poissrnd(r);
        elseif size(param) == [344 252]% Sss simulation, need activty image and attenuation.
            if strcmpi(objGpet.method_for_randoms,'e7_tools')
                % Call e7 tools:
                error(sprintf('Randoms with %s not available for 2d.', objGpet.method_for_randoms));
            elseif strcmpi(objGpet.method_for_randoms,'from_ML_singles_stir')
                error(sprintf('Randoms with %s not available for 2d.', objGpet.method_for_randoms));
            elseif strcmpi(objGpet.method_for_randoms,'from_ML_singles_matlab')
                structSizeSino2d = getSizeSino2dStruct(objGpet.sinogram_size.nRadialBins, objGpet.sinogram_size.nAnglesBins, ...
                    objGpet.sinogram_size.nRings, 596/2, 260);
                % Call matlab function in apirl:
                numIterations = 3;
                [r, singlesOut] = estimateRandomsFromDelayeds2d(varargin{2}, structSizeSino2d, numIterations);
            end
        end
    end
 end