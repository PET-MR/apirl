function [Img,totalScaleFactor, info] = BQML(objGpet,Img,sinogramFilename,normalizationFilename)

info.S = getInfoFromInterfile(sinogramFilename);
info.N = getInfoFromInterfile(normalizationFilename);


proportionality_factor =  1.05; % Obtained by ROI-based SUV comparsions with e7 DICOM images

counts_per_voxel = objGpet.image_size.matrixSize(1)/objGpet.sinogram_size.nRadialBins;

corrected_pixel_size = objGpet.scanner_properties.binSize_mm ; % need to figure it out 

LOR_DOI_correction = (objGpet.scanner_properties.radius_mm + objGpet.scanner_properties.sinogramDepthOfInteraction_mm)/ objGpet.scanner_properties.radius_mm ;

decay_correction_factor = decay_factor(info.S.ImageRelativeStartTimeSec, info.S.ImageDurationSec, info.S.IsotopeGammaHalflifeSec);

frame_length_correction = 1.0 / info.S.ImageDurationSec;

scale_factor = frame_length_correction .* decay_correction_factor;

loss_correction_factors = info.S.GimLossFraction .* info.S.PdrLossFraction;

totalScaleFactor = proportionality_factor.*(info.N.ScannerQuantificationFactorBqSEcatCounts*info.S.IsotopeBranchingFactor).*...
    scale_factor.*LOR_DOI_correction *counts_per_voxel * corrected_pixel_size.*loss_correction_factors ;%

Img = Img.*totalScaleFactor;

function decay_factor1 = decay_factor(frame_start,frame_duration,thalf)


test = frame_duration/thalf;

if (test < 0.3)
    ln2 = log(2.0);
    lt1 = frame_duration * ln2 / thalf;
    decay_factor1 = exp(frame_start * ln2 / thalf) / double((1.0-lt1/2.0)...
        + (lt1*lt1/6.0) ...
        - (lt1*lt1*lt1/24.0) ...
        + (lt1*lt1*lt1*lt1/120.0) ...
        - (lt1*lt1*lt1*lt1*lt1/720.0));
else
    lambda = -log(0.5)/thalf;
    b1 = exp(lambda*frame_start);
    b2 = (frame_duration*lambda)/(1-exp(-lambda*frame_duration)) ;
    decay_factor1 = b1*b2 ;
end
