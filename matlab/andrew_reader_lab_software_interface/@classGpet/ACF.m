% *********************************************************************
% Reconstruction Framework for Siemens Biograph mMR.  
% class: Gpet
% Authors: Martin Belzunce, Abolfazl Mehranian. Kings College London.
% Date: 08/02/2016
% *********************************************************************
% Generate attenuation correction factors.
function a=ACF(objGpet, attenuationMap_1_cm, refImage)
    if ~strcmpi(objGpet.scanner,'mMR')&& ~strcmpi(objGpet.scanner,'cylindrical')
     error('ACFs are only available for mMR scanner.');
    end
    if strcmpi(objGpet.scanner,'mMR')
        pixelSize_mm = [refImage.PixelExtentInWorldY refImage.PixelExtentInWorldX refImage.PixelExtentInWorldZ];
        if strcmpi(objGpet.method, 'otf_siddon_cpu')
            structSizeSino3d = getSizeSino3dFromSpan(objGpet.sinogram_size.nRadialBins, objGpet.sinogram_size.nAnglesBins, ...
                objGpet.sinogram_size.nRings, 596/2, 260, ...
                objGpet.sinogram_size.span, objGpet.sinogram_size.maxRingDifference);
            a = createACFsFromImage(attenuationMap_1_cm, pixelSize_mm, objGpet.tempPath, 'acf', structSizeSino3d, 0, 0);
        elseif strcmpi(objGpet.method, 'otf_siddon_gpu')
            structSizeSino3d = getSizeSino3dFromSpan(objGpet.sinogram_size.nRadialBins, objGpet.sinogram_size.nAnglesBins, ...
                objGpet.sinogram_size.nRings, 596/2, 260, ...
                objGpet.sinogram_size.span, objGpet.sinogram_size.maxRingDifference);
            a = createACFsFromImage(attenuationMap_1_cm, pixelSize_mm, objGpet.tempPath, 'acf', structSizeSino3d, 0, 1);
        end
    elseif strcmpi(objGpet.scanner,'cylindrical')
        pixelSize_mm = [refImage.PixelExtentInWorldY refImage.PixelExtentInWorldX refImage.PixelExtentInWorldZ];
        structSizeSino = get_sinogram_size_for_apirl(objGpet);
        if strcmpi(objGpet.method, 'otf_siddon_cpu')
            a = createACFsFromImage(attenuationMap_1_cm, pixelSize_mm, objGpet.tempPath, 'acf', structSizeSino, 0, 0, objGpet.scanner, objGpet.scanner_properties);
        elseif strcmpi(objGpet.method, 'otf_siddon_gpu')
            a = createACFsFromImage(attenuationMap_1_cm, pixelSize_mm, objGpet.tempPath, 'acf', structSizeSino, 0, 1, objGpet.scanner, objGpet.scanner_properties);
        end
    end
end