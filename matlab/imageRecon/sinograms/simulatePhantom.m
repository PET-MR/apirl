% phantom and attenuationMap must have the same size.
% refImage has the matrix size and pixel size of both images.
% span or structSizeSino is the size of the sinogram to simulate.
% averageCountsInSino is the average counts in the sinogram to introduce
% the poisson noise.
% normFactors is the normalization factors in a sinogram with
% structSizeSino size or with the filename of a mmr .n file.
% randomsFraction is the porcentaje of randoms.
% scatterFraction is the scatter fraction in the sinogram.
% outputPath for temporary files.

function [emissionSinogram, afsSinogram, randomsSinogram, scatterSinogram] = simulatePhantom(phantom, attenuationMap, refImage, span, averageCountsInSino, normFactors, randomsFraction, scatterFraction, outputPath, useGpu)

if ~isdir(outputPath)
    mkdir(outputPath)
end

pixelSize_mm = [refImage.PixelExtentInWorldX refImage.PixelExtentInWorldY refImage.PixelExtentInWorldZ];
imageSize_pixels = size(phantom);

% Generat a noisy phantom that simualtes the random poisson emission from
% each voxel. 
meanPhantom = mean(mean(mean(phantom(phantom>0))));

%% EMISSION DISTRIBUTION
% Project the phantom to generate the emission sinogram:
[projectedSinogram, structSizeSino3d] = ProjectMmr(phantom, pixelSize_mm, outputPath, span, [], [], useGpu);
%% NORMALIZATION FACTORS
if isstr(normFactors)
    disp('Computing the normalization correction factors...');
    % ncf:
    [overall_ncf_3d, scanner_time_invariant_ncf_3d, scanner_time_variant_ncf_3d, acquisition_dependant_ncf_3d, used_xtal_efficiencies, used_deadtimefactors, used_axial_factors] = ...
       create_norm_files_mmr(normFactors, [], [], [], [], structSizeSino3d.span);
    % invert for nf:
    overall_nf_3d = overall_ncf_3d;
    overall_nf_3d(overall_ncf_3d ~= 0) = 1./overall_nf_3d(overall_ncf_3d ~= 0);
else
    if ~isempty(normFactors)   % if empty no norm is used, if not used the matrix in normFilename proving is of the same size of the sinogram.
        if size(normFactors) ~= size(projectedSinogram)
            error('The size of the normalization correction factors is incorrect.')
        end
        disp('Using the normalization correction factors received as a parameter...');
        overall_ncf_3d = normFactors;
        clear normFilename;
        % invert for nf:
        overall_nf_3d = overall_ncf_3d;
        overall_nf_3d(overall_ncf_3d ~= 0) = 1./overall_nf_3d(overall_ncf_3d ~= 0);
    else
        overall_ncf_3d = ones(size(projectedSinogram));
        overall_nf_3d = overall_ncf_3d;
    end
end
%% ATTENUATION FACTORS
if ~isempty(attenuationMap)
    % Generate acfs:
    acfsSinogram = createACFsFromImage(attenuationMap, pixelSize_mm, outputPath, 'acfs', structSizeSino3d, 0, useGpu);
    afsSinogram = acfsSinogram;
    afsSinogram(afsSinogram~=0) = 1./afsSinogram(afsSinogram~=0);
    clear acfsSinogram;
else
    afsSinogram = ones(size(projectedSinogram));
end
%% RANDOMS
randomsSinogram = zeros(size(projectedSinogram));
if ~isempty(randomsFraction)
    if randomsFraction ~= 0
        
    end
end
%% SCATTER
scatterSinogram = zeros(size(projectedSinogram));
if ~isempty(scatterFraction)
    if scatterFraction ~= 0
        
    end
end
%% COMPLETE FORWARD MODEL
emissionSinogram = projectedSinogram.* afsSinogram .*overall_nf_3d + randomsSinogram .* overall_nf_3d + scatterSinogram;
%% FINALLY, INTRODUCE POISSON NOISE
% Get the mean values for the projected sinogram:
meanProjectedPhantom = mean(mean(mean(emissionSinogram(emissionSinogram>0))));
% Apply eh scaling to the phantom and generate random counts:
emissionSinogram = poissrnd(emissionSinogram.*averageCountsInSino./ meanProjectedPhantom);
% % After generating the noise, rescale again to get the same mean
% % value of each simulated phantom:
% noisySinogram = noisySinogram ./ scalingRatio;