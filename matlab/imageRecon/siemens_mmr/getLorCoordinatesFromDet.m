%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 18/04/2016
%  *********************************************************************
%  function [coordP1 coordP2] = getLorCoordinatesFromDet(indTheta,indR, indZ)
%
%  This functions returns the geomtrical coordinates for a LOR from the
%  index of the theta angle, distance r and plane of the sinogram bin.

function [coordX1, coordY1, coordZ1, coordX2, coordY2, coordZ2, r, theta_deg] = getLorCoordinatesFromDet(indTheta,indR, indZ1,indZ2)

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
r_without_arc_correc_mm = binSize_mm.*([0:numR-1]-numR/2); % 
% ad hoc correction:
rValues_mm = (radioScanner_mm + meanDOI_mm .* cos(r_without_arc_correc_mm./radioScanner_mm)) .* sin(r_without_arc_correc_mm./(radioScanner_mm));
%rValues_mm(thetaValues_deg>90) = rValues_mm(thetaValues_deg>90) - binSize_mm/2;
effRadioScanner_mm = (radioScanner_mm + meanDOI_mm);
% axial values:
ptrAxialvalues_mm = widthRings_mm/2 + widthRings_mm*[0:numRings-1];

% Can be a simple indes or a range of values:
[iR, iTheta, iZ1, iZ2] = ndgrid(indR, indTheta, indZ1, indZ2);

% Correct half angles (odd r indices have the correct angle, even r have half angles):
angle_offset_deg = -rem(iR+1,2).*(deltaTheta_deg/2);
angle_offset = deg2rad(angle_offset_deg);
% Get coordonates fo the points:
coordX1 = [rValues_mm(iR).*cos(thetaValues_rad(iTheta)+angle_offset) + sin(thetaValues_rad(iTheta)+angle_offset) .* sqrt(effRadioScanner_mm.^2-rValues_mm(iR).^2)];
coordY1 = [rValues_mm(iR).*sin(thetaValues_rad(iTheta)+angle_offset) - cos(thetaValues_rad(iTheta)+angle_offset) .* sqrt(effRadioScanner_mm.^2-rValues_mm(iR).^2)];
coordZ1 = [ptrAxialvalues_mm(iZ1) - meanDOI_mm./(effRadioScanner_mm) .* ( ptrAxialvalues_mm(iZ2)- ptrAxialvalues_mm(iZ1))];
coordX2 = [rValues_mm(iR).*cos(thetaValues_rad(iTheta)+angle_offset) - sin(thetaValues_rad(iTheta)+angle_offset) .* sqrt(effRadioScanner_mm.^2-rValues_mm(iR).^2)];
coordY2 = [rValues_mm(iR).*sin(thetaValues_rad(iTheta)+angle_offset) + cos(thetaValues_rad(iTheta)+angle_offset) .* sqrt(effRadioScanner_mm.^2-rValues_mm(iR).^2)];
coordZ2 = [ptrAxialvalues_mm(iZ2) - meanDOI_mm./(effRadioScanner_mm) .* ( ptrAxialvalues_mm(iZ1)- ptrAxialvalues_mm(iZ2))];
r = rValues_mm(iR);
theta_deg = thetaValues_deg(iTheta)+angle_offset_deg;








