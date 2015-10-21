%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 19/02/2015
%  *********************************************************************
%  function [mapaDet1Ids mapaDet2Ids] = createMmrDetectorsIdInSinogram()
%
%  This functions create two sinograms with the id of each detector in a
%  bin LOR.

function [mapaDet1Ids, mapaDet2Ids] = createMmrDetectorsIdInSinogram()

% Size of mMr Sinogram's
numTheta = 252; numR = 344; numRings = 64; maxAbsRingDiff = 60; rFov_mm = 594/2; zFov_mm = 258; span = 1;
structSizeSino3d = getSizeSino3dFromSpan(numR, numTheta, numRings, rFov_mm, zFov_mm, span, maxAbsRingDiff);

% First we create a map with the indexes fo each crystal element in each
% transverse 2d sinogram.
mapaDet1Ids = zeros(structSizeSino3d.numR, structSizeSino3d.numTheta, 'uint16');
mapaDet2Ids = zeros(structSizeSino3d.numR, structSizeSino3d.numTheta, 'uint16');
numDetectors = 504;

% Create the map with the id of the crystal element for detector 1 and
% detector 2:
theta = [0:structSizeSino3d.numTheta-1]'; % The index of thetas goes from 0 to numTheta-1 (in stir)
r = (-structSizeSino3d.numR/2):(-structSizeSino3d.numR/2+structSizeSino3d.numR-1);
[THETA, R] = meshgrid(theta,r);
mapaDet1Ids = rem((THETA + floor(R/2) + numDetectors -1), numDetectors) + 1;   % The +1 is added in matlab version respect than c version, because here we have 1-base indexes.
mapaDet2Ids = rem((THETA - floor((R+1)/2) + numDetectors/2 -1), numDetectors) + 1; % The +1 is added in matlab version respect than c version, because here we have 1-base indexes.
