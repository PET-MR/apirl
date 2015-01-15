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
function [sinograms, delayedSinograms] = getIntfSinogramsFromUncompressedMmr(filenameUncompressedMmr, outFilenameIntfSinograms)

% Size of mMr Sinogram's
numTheta = 252; numR = 344; numRings = 64; maxAbsRingDiff = 60; rFov_mm = 594; zFov_mm = 258; span = 1;
structSizeSino3d = getSizeSino3dFromSpan(numTheta, numR, numRings, rFov_mm, zFov_mm, span, maxAbsRingDiff);
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

% Rearrange in a 3d matrix:
sinograms = reshape(sinograms, [  numR numTheta numSinograms]);
delayedSinograms = reshape(delayedSinograms, [  numR numTheta numSinograms]);
% Matlab reads in a column-wise order that why angles are in the columns.
% We want to have it in the rows since APIRL and STIR and other libraries
% use row-wise order:
sinograms = permute(sinograms,[2 1 3]);
sinograms = permute(delayedSinograms,[2 1 3]);
% Wirte the interfile prompt sinogram:
interfileWriteSino(sinograms, outFilenameIntfSinograms, structSizeSino3d.sinogramsPerSegment, structSizeSino3d.minRingDiff, structSizeSino3d.maxRingDiff);
% Wirte the interfile delayed sinogram:
interfileWriteSino(delayedSinograms, [outFilenameIntfSinograms '_delay'], structSizeSino3d.sinogramsPerSegment, structSizeSino3d.minRingDiff, structSizeSino3d.maxRingDiff);
