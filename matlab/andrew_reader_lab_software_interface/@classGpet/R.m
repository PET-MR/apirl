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
    if ~strcmpi(objGpet.scanner,'mMR')&& ~strcmpi(objGpet.scanner,'cylindrical')
        error('Randoms are only available for mMR or cylindrical scanner.');
    end
    if nargin == 2
        param = varargin{2};
    end
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
                structSizeSino = get_sinogram_size_for_apirl(objGpet);
                if isfield(structSizeSino, 'sinogramsPerSegment')
                    if numel(structSizeSino.sinogramsPerSegment) == 1
                        if structSizeSino.numZ == 1
                             numIterations = 3;
                             [r, singlesOut] = estimateRandomsFromDelayeds2d(varargin{2}, structSizeSino, numIterations);
                        else
                             numIterations = 3;
                             [r, singlesOut] = estimateRandomsFromDelayeds2d(varargin{2}, structSizeSino, numIterations);
                        end
                    else
                        % Call matlab function in apirl:
                        numIterations = 3;
                        [r, singlesOut] = estimateRandomsFromDelayeds(varargin{2}, structSizeSino, numIterations, objGpet.sinogram_size.span);
                    end
                else
                    if structSizeSino.numZ == 1
                        numIterations = 3;
                        [r, singlesOut] = estimateRandomsFromDelayeds2d(varargin{2}, structSizeSino, numIterations);
                    else
                        numIterations = 3;
                        [r, singlesOut] = estimateRandomsFromDelayeds2d(varargin{2}, structSizeSino, numIterations);
                    end
                end
            end
        end
    elseif strcmpi(objGpet.scanner,'cylindrical')
        if numel(param) == 1 % Simple simulation, constant background with poisson dsitribution, the input parameter is the mean total number of counts.
            counts = varargin{2};
            r = ones(objGpet.sinogram_size.matrixSize);
            meanValue = counts./numel(r);
            r = r .* meanValue;
            % Generate a poisson distributed with contant mean value:
            r =poissrnd(r);
        elseif strcmpi(objGpet.method_for_randoms,'from_ML_singles_matlab')
            error('Randoms using from_ML_singles_matlab are only available for mMR scanner.');
        
        end
    end
 end