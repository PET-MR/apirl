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
    pixelSize_mm = [refImage.PixelExtentInWorldY refImage.PixelExtentInWorldX refImage.PixelExtentInWorldZ];
    % if span 0 the attenuation map needs to have the same number of slcies
    % than rings:
    if objGpet.sinogram_size.span == 0 && (objGpet.sinogram_size.nRings ~= size(attenuationMap_1_cm,3))
        rings = 1 : objGpet.sinogram_size.nRings;
        slices = 1 : objGpet.sinogram_size.nRings/(size(attenuationMap_1_cm,3)+1) : objGpet.sinogram_size.nRings;
        [X1,Y1,Z1] = meshgrid(1:size(attenuationMap_1_cm,2),1:size(attenuationMap_1_cm,1),slices);
        [X2,Y2,Z2] = meshgrid(1:size(attenuationMap_1_cm,2),1:size(attenuationMap_1_cm,1),rings);
        attenuationMap_1_cm = interp3(X1,Y1,Z1,attenuationMap_1_cm,X2,Y2,Z2);
        pixelSize_mm = [refImage.PixelExtentInWorldY refImage.PixelExtentInWorldX refImage.PixelExtentInWorldZ*refImage.ImageSize(3)./size(attenuationMap_1_cm,3)];
    end
    if strcmpi(objGpet.scanner,'mMR')        
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
        structSizeSino = get_sinogram_size_for_apirl(objGpet);
        if strcmpi(objGpet.method, 'otf_siddon_cpu')
            a = createACFsFromImage(attenuationMap_1_cm, pixelSize_mm, objGpet.tempPath, 'acf', structSizeSino, 0, 0, objGpet.scanner, objGpet.scanner_properties);
        elseif strcmpi(objGpet.method, 'otf_siddon_gpu')
            a = createACFsFromImage(attenuationMap_1_cm, pixelSize_mm, objGpet.tempPath, 'acf', structSizeSino, 0, 1, objGpet.scanner, objGpet.scanner_properties);
        end
    end
end