%% FUNCTION THAT CREATES ATTENUATION MAP FROM A CT SCAN
% Receives the path of the ct in dicom format.
% Return muMap511FromCt
function muMap511FromCt = CreateAttenuationMapsFromCt(imageCt, RescaleSlope, RescaleIntercept, KVP, showConversionCurve)

% Rescale to HU:
imageCt = imageCt.*RescaleSlope + RescaleIntercept;

% CONVERT CT IN MU MAP AT 511 KEV
% We use the method with two linear conversions, one slope for HU of 0 or
% less (soft tissue) and another one for HU greater than 0 (bones).
% The correction factor for higher density tissues depends of the 140 kVp
% (Bai 2003):
switch(KVP)
    case 140
        corrConvFactorHigherDensity = 0.640;
    case 130
        corrConvFactorHigherDensity = 0.605;
    case 120
        corrConvFactorHigherDensity = 0.576;
    case 110
        corrConvFactorHigherDensity = 0.509;
    otherwise
        corrConvFactorHigherDensity = 0.509;
end

% Plot the conversion curve:
valuesHU = -1000 : 2000;
linAttenCoefWater_1_cm = 0.096;
conversionFactosHUtoMu511 = [(1+valuesHU(valuesHU<=0)/1000).*linAttenCoefWater_1_cm (1+corrConvFactorHigherDensity.*valuesHU(valuesHU>0)/1000).*linAttenCoefWater_1_cm];
% Conversion factor for siemens:
switch KVP
    case 80,  v =[9.5e-005, 3.64e-005, 0.0626, 1050];
    case 100, v =[9.5e-005, 4.43e-005, 0.0544, 1052];
    case 110, v =[9.5e-005, 4.63e-005, 0.0521, 1043];
    case 120, v =[9.5e-005, 5.1e-005, 0.0471, 1047];
    case 130, v =[9.5e-005, 5.21e-005, 0.0457, 1037];
    case 140, v =[9.5e-005, 5.64e-005, 0.0408, 1030];
    otherwise, v =[9.5e-005, 5.1e-005, 0.0471, 1050];
end
conversionFactosHUtoMu511_cti = valuesHU +1000;
bp = conversionFactosHUtoMu511_cti < v(4);
conversionFactosHUtoMu511_cti(bp) = v(1)*conversionFactosHUtoMu511_cti(bp);
conversionFactosHUtoMu511_cti(~bp) = v(2)*conversionFactosHUtoMu511_cti(~bp)+v(3);
mu = max(0,conversionFactosHUtoMu511_cti);
% Plot to compare:
if showConversionCurve
    figure;
    plot(valuesHU,conversionFactosHUtoMu511, valuesHU,conversionFactosHUtoMu511_cti, 'LineWidth',3); hold on;
    ylabel('Conversion Factor');
    xlabel('Hounsfield Units');
    legend('my implementation of bai 2003','e7')
end
% Apply the conversion to the ct image:
softTissue = imageCt <= 0;
boneTissue = imageCt > 0;
muMap511FromCt = zeros(size(imageCt));
muMap511FromCt(softTissue) = (1+imageCt(softTissue)/1000).*0.096;
muMap511FromCt(boneTissue) = (1+corrConvFactorHigherDensity.*imageCt(boneTissue)/1000).*0.096;
muMap511FromCt(muMap511FromCt<0) = 0;