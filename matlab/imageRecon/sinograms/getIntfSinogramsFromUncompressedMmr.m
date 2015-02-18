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
function [sinograms, delayedSinograms, structSizeSino3d] = getIntfSinogramsFromUncompressedMmr(filenameUncompressedMmr, outFilenameIntfSinograms)

% Size of mMr Sinogram's
numTheta = 252; numR = 344; numRings = 64; maxAbsRingDiff = 60; rFov_mm = 594/2; zFov_mm = 258; span = 1;
structSizeSino3d = getSizeSino3dFromSpan(numR, numTheta, numRings, rFov_mm, zFov_mm, span, maxAbsRingDiff);
% Total number of singorams per 3d sinogram:
numSinograms = sum(structSizeSino3d.sinogramsPerSegment);

% Read raw data from uncomrpessed sinogram:
[fid, message] = fopen(filenameUncompressedMmr,'r');
if fid == -1
    disp(ferror(fid));
end
% First the prompts:
[sinograms, count] = fread(fid, numTheta*numR*numSinograms, 'int16=>int16');
% Then the delayed:
[delayedSinograms, count] = fread(fid, numTheta*numR*numSinograms, 'int16=>int16');
% Close the file:
fclose(fid);

%Convert to single:
sinograms = single(sinograms);
delayedSinograms = single(delayedSinograms);
% Rearrange in a 3d matrix:
sinograms = reshape(sinograms, [numR numTheta numSinograms]);
delayedSinograms = reshape(delayedSinograms, [numR numTheta numSinograms]);
% Wirte the interfile prompt sinogram:
interfileWriteSino(sinograms, outFilenameIntfSinograms, structSizeSino3d);
% Wirte the interfile delayed sinogram:
interfileWriteSino(delayedSinograms, [outFilenameIntfSinograms '_delay'], structSizeSino3d);
