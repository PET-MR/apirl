% *********************************************************************
% Reconstruction Framework for Siemens Biograph mMR.  
% class: Gpet
% Authors: Martin Belzunce, Abolfazl Mehranian. Kings College London.
% Date: 08/02/2016
% *********************************************************************
% Generate attenuation correction factors.
function a=ACF(objGpet, attenuationMap_1_cm, refImage)
    pixelSize_mm = [refImage.PixelExtentInWorldY refImage.PixelExtentInWorldX refImage.PixelExtentInWorldZ];
    if ~strcmpi(objGpet.scanner,'mMR')
     error('ACFs are only available for mMR scanner.');
    end
    if strcmpi(objGpet.method, 'otf_siddon_cpu')
        structSizeSino3d = getSizeSino3dFromSpan(objGpet.sinogram_size.nRadialBins, objGpet.sinogram_size.nAnglesBins, ...
            objGpet.sinogram_size.nRings, objGpet.sinogram_size.rFov_mm, objGpet.sinogram_size.zFov_mm, ...
            objGpet.sinogram_size.span, objGpet.sinogram_size.maxRingDifference);
        a = createACFsFromImage(attenuationMap_1_cm, pixelSize_mm, objGpet.tempPath, 'acf', structSizeSino3d, 0, useGpu);
    elseif strcmpi(objGpet.method, 'otf_siddon_gpu')
        structSizeSino3d = getSizeSino3dFromSpan(objGpet.sinogram_size.nRadialBins, objGpet.sinogram_size.nAnglesBins, ...
            objGpet.sinogram_size.nRings, 596/2, 260, ...
            objGpet.sinogram_size.span, objGpet.sinogram_size.maxRingDifference);
        a = createACFsFromImage(attenuationMap_1_cm, pixelSize_mm, objGpet.tempPath, 'acf', structSizeSino3d, 0, 1);
    end
end