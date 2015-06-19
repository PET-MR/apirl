%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 15/01/2015
%  *********************************************************************
%  function [sinogram, delayedSinogram] = getIntfSinogramsFromUncompressedMmr(filenameUncompressedMmr, outFilenameIntfSinograms)
% 
%  This functions read an uncompressed interfile sinogram from a Siemens
%  Biograph mMr acquistion, where both promtps and randosm are included and
%  creates interfile sinograms to be used in APIRL reconstruction and it
%  might be used in STIR also. The size of the sinograms are fixed to
%  Siemens Biograph mMr standard configuration.
%  Size of standard mMr Sinogram's:
%  numTheta = 252; numR = 344; numRings = 64; maxAbsRingDiff = 60; rFov_mm = 594; zFov_mm = 258; span = 1;
%  filenameUncompressedMmr : filename of the binaty file with th mMr
%  uncompressed acquisition data. It should includes the prompts and
%  delayed sinograms.
% 
%  outFilenameIntfSinograms: name for the output interfile prompt sinogram,
%  the delayed hast the label '_delayed' at the end.
%
%  It returns the sinogram, the delayed sinogram and an struct with the
%  sinogram size.
function [subset, structSizeSino3dSubset] = getSubsetFromSinogram(sinogram, structSizeSino3dComplete, numberOfSubsets, subsetIndex)

% Size of the subset:
structSizeSino3dSubset = structSizeSino3dComplete;
structSizeSino3dSubset.numTheta = ceil(structSizeSino3dComplete.numTheta/numberOfSubsets);

% Total number of singorams per 3d sinogram:
numSinograms = sum(structSizeSino3dComplete.sinogramsPerSegment);

% Generate the subset:
subset = zeros(structSizeSino3dSubset.numR, structSizeSino3dSubset.numTheta, numSinograms);

% Fill the subset:
subset = sinogram(:, subsetIndex : numberOfSubsets : end, :);


