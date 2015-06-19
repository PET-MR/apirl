%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 10/06/2015
%  *********************************************************************
%  function [sinogramSpanN, structSizeSino3dSpanN] = convertSinogramToSpan(sinogram, structSizeSino3d, outputSpan)
% 
%  This functions get a sinogram of size defined in the input parameter
%  structSizeSino3d and converts to a desired span (outputSpan).
%
function [sinogramSpanN, structSizeSino3dSpanN] = convertSinogramToSpan(sinogram, structSizeSino3d, outputSpan)


% Create sinogram span N:
michelogram = generateMichelogramFromSinogram3D(sinogram, structSizeSino3d);
structSizeSino3dSpanN = getSizeSino3dFromSpan(structSizeSino3d.numR, structSizeSino3d.numTheta, structSizeSino3d.numZ, ...
    structSizeSino3d.rFov_mm, structSizeSino3d.zFov_mm, outputSpan, structSizeSino3d.maxAbsRingDiff);
sinogramSpanN = reduceMichelogram(michelogram, structSizeSino3dSpanN);

