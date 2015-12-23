%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 30/10/2015
%  *********************************************************************
%  [sinogramReduced, structSizeSino3dReduced] = reduceSinogramSize(sinogram, structSizeSino3d, outputNumRings)
% 
%  This functions get reduce a sinogram to a lower version.
%
function [sinogramReduced, structSizeSino3dReduced] = reduceSinogramSize(sinogram, structSizeSino3d, structSizeSino3dReduced)


% Create sinogram span N:
michelogram = generateMichelogramFromSinogram3D(sinogram, structSizeSino3d);
structSizeSino3dSpanN = getSizeSino3dFromSpan(structSizeSino3d.numR, structSizeSino3d.numTheta, structSizeSino3d.numZ, ...
    structSizeSino3d.rFov_mm, structSizeSino3d.zFov_mm, outputSpan, structSizeSino3d.maxAbsRingDiff);
sinogramSpanN = reduceMichelogram(michelogram, structSizeSino3dSpanN);

