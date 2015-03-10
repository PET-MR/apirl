%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 25/02/2015
%  *********************************************************************
%  function convertMmrIntfSinogramToSpan(filenameUncompressedMmr, outFilenameIntfSinograms, span)
% 
%  This functions read an uncompressed interfile sinogram from a Siemens
%  Biograph mMr acquistion, where both promtps and randosm are included and
%  creates interfile sinograms to be used in APIRL reconstruction and it
%  might be used in STIR also. The size of the sinograms are fixed to
%  Siemens Biograph mMr standard configuration. It copies the header and
%  replace the fields that must be changed.
%  Size of standard mMr Sinogram's:
%  numTheta = 252; numR = 344; numRings = 64; maxAbsRingDiff = 60; rFov_mm = 594; zFov_mm = 258; span = 1;
%  filenameInterfileMmr : filename of the header file with the mMr
%  uncompressed acquisition data. It should includes the prompts and
%  delayed sinograms.
% 
%  outFilenameIntfSinograms: name for the output interfile prompt sinogram,
%  the delayed hast the label '_delayed' at the end. It's the name without
%  any extension, then a .s and .hs is generated (to be used with stir),
%  and also an .h33 and .i33 to be used with APIRL.
%
%  span: span of the ouput sinogram.
%
%  Return structSizeSino3dSpanN, the size of the sinogram. 
%
function structSizeSino3dSpanN = convertMmrIntfSinogramToSpan(filenameUncompressedMmr, outFilenameIntfSinograms, span)

% Size of mMr Sinogram's
[structInterfile, structSizeSino] = getInfoFromSiemensIntf(filenameUncompressedMmr);
% Total number of singorams per 3d sinogram:
numSinograms = sum(structSizeSino.sinogramsPerSegment);

% Get filename of raw data:
filenameBinary = structInterfile.NameOfDataFile;

% Read raw data from uncomrpessed sinogram:
[fid, message] = fopen(filenameBinary,'r');
if fid == -1
    disp(ferror(fid));
end
% First the prompts:
[sinograms, count] = fread(fid, structSizeSino.numTheta*structSizeSino.numR*numSinograms, 'int16=>int16');
% Then the delayed:
[delayedSinograms, count] = fread(fid, structSizeSino.numTheta*structSizeSino.numR*numSinograms, 'int16=>int16');
% Close the file:
fclose(fid);

%Convert to single:
sinograms = single(sinograms);
delayedSinograms = single(delayedSinograms);
% Rearrange in a 3d matrix:
sinograms = reshape(sinograms, [structSizeSino.numR structSizeSino.numTheta numSinograms]);
delayedSinograms = reshape(delayedSinograms, [structSizeSino.numR structSizeSino.numTheta numSinograms]);

%% Outupur for APIRL
% Create sinogram span N:
michelogram = generateMichelogramFromSinogram3D(sinograms, structSizeSino);
structSizeSino3dSpanN = getSizeSino3dFromSpan(structSizeSino.numR, structSizeSino.numTheta, structSizeSino.numZ, ...
    structSizeSino.rFov_mm, structSizeSino.zFov_mm, span, structSizeSino.maxAbsRingDiff);
sinogramSpanN = reduceMichelogram(michelogram, structSizeSino3dSpanN);
% Wirte the interfile prompt sinogram:
interfileWriteSino(sinogramSpanN, outFilenameIntfSinograms, structSizeSino3dSpanN);

% The same for delayed:
michelogram = generateMichelogramFromSinogram3D(delayedSinograms, structSizeSino);
dealyedSinogramSpanN = reduceMichelogram(michelogram, structSizeSino3dSpanN);
% Wirte the interfile delayed sinogram:
interfileWriteSino(dealyedSinogramSpanN, [outFilenameIntfSinograms '_delay'], structSizeSino3dSpanN);

%% Output for Stir (Keeps the header but replaces the fields that changes)
% Write using the funcition to write interfile sinograms with the stir
% interfile format:
interfileWriteStirSino(sinogramSpanN, outFilenameIntfSinograms, structSizeSino3dSpanN);
interfileWriteStirSino(dealyedSinogramSpanN, [outFilenameIntfSinograms '_delay'], structSizeSino3dSpanN);

