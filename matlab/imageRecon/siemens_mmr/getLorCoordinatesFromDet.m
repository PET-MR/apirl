%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 18/04/2016
%  *********************************************************************
%  function [coordP1 coordP2] = getLorCoordinatesFromDet(indTheta,indR, indZ)
%
%  This functions returns the geomtrical coordinates for a LOR from the
%  index of the theta angle, distance r and plane of the sinogram bin.

function [coordP1, coordP2] = getLorCoordinatesFromDet(indTheta,indR, indZ1,indZ2)

% Scanner constants:
crystalElementSize_mm = 4.0891;
binSize_mm = 4.0891/2.0;
crystalElementLength_mm = 20;
widthRings_mm = 4.0571289;
meanDOI_mm = 9.6;
radioScanner_mm = 328;
radioFov_mm = 297;
numTheta = 252;
numR = 344;
numRings = 64;

% Angles:
deltaTheta_deg = 180/numTheta;
thetaValues_deg = [0:numTheta-1].*deltaTheta_deg;
thetaValues_rad = deg2rad(thetaValues_deg);
% R:
r_without_arc_correc_mm = binSize_mm/2 + binSize_mm.*([0:numR-1]-numR/2);
rValues_mm = (radioScanner_mm + meanDOI_mm .* cos(r_without_arc_correc_mm./radioScanner_mm)) .* sin(r_without_arc_correc_mm./(radioScanner_mm));
effRadioScanner_mm = (radioScanner_mm + meanDOI_mm* cos(r_without_arc_correc_mm/radioScanner_mm));
% axial values:
ptrAxialvalues_mm = widthRings_mm/2 + widthRings_mm*[0:numRings-1];


% Get coordonates fo the points:
coordP1 = [rValues_mm(indR).*cos(thetaValues_rad(indTheta)) + sin(thetaValues_rad(indTheta)) .* sqrt(effRadioScanner_mm(indR).^2-rValues_mm(indR).^2); ...
    rValues_mm(indR).*sin(thetaValues_rad(indTheta)) - cos(thetaValues_rad(indTheta)) .* sqrt(effRadioScanner_mm(indR).^2-rValues_mm(indR).^2); ...
    ptrAxialvalues_mm(indZ1) - meanDOI_mm./(effRadioScanner_mm(indR)) .* ( ptrAxialvalues_mm(indZ2)- ptrAxialvalues_mm(indZ1))]';
coordP2 = [rValues_mm(indR).*cos(thetaValues_rad(indTheta)) - sin(thetaValues_rad(indTheta)) .* sqrt(effRadioScanner_mm(indR).^2-rValues_mm(indR).^2); ...
    rValues_mm(indR).*sin(thetaValues_rad(indTheta)) + cos(thetaValues_rad(indTheta)) .* sqrt(effRadioScanner_mm(indR).^2-rValues_mm(indR).^2); ...
    ptrAxialvalues_mm(indZ2) - meanDOI_mm./(effRadioScanner_mm(indR)) .* ( ptrAxialvalues_mm(indZ1)- ptrAxialvalues_mm(indZ2))]';








