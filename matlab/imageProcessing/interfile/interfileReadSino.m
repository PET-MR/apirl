%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 10/06/2015
%  *********************************************************************
%  function [sinograms, delayedSinograms, structSizeSino3d] = interfileReadSino(filenameHeader)
% 
%  This functions read interfile sinograms, for both APIRL and uncompressed
%  interfile sinogram from a Siemens Biograph mMr acquistion. The latter
%  includes both promtps and randosm. For the apirl, the delayed sinograms
%  is returned empty. The size of the sinogram is obtained from the header.
%
%  It returns the sinogram, the delayed sinogram and an struct with the
%  sinogram size.
function [sinograms, delayedSinograms, structSizeSino] = interfileReadSino(filenameHeader)

[info, structSizeSino] = getInfoFromInterfile(filenameHeader);

if strcmp(info.NumberFormat, 'short float') || strcmp(info.NumberFormat, 'float')
    readingFormat = 'single=>single';
elseif  strcmp(info.NumberFormat, 'signed integer')
        if info.NumberOfBytesPerPixel == 2
            readingFormat = 'int16=>int16';
        elseif info.NumberOfBytesPerPixel == 4
            readingFormat = 'int32=>int32';
        end
end
if isfield(info, 'sinogramsPerSegment')||isfield(info, 'ScanDataTypeDescription2')  % If is a siemens interfile or a 3d intefile sinogram.
    numSinograms = sum(structSizeSino.sinogramsPerSegment);
else
    numSinograms = structSizeSino.numZ; % sinogram2d.
end

% Check if is an siemens sinogram (mmr):
if isfield(info, 'ScanDataTypeDescription2')
    % Read raw data from uncomrpessed sinogram:
    [fid, message] = fopen(info.NameOfDataFile,'r');
    if fid == -1
        disp(ferror(fid));
    end
    % First the prompts:
    [sinograms, count] = fread(fid, structSizeSino.numTheta*structSizeSino.numR*numSinograms, readingFormat);
    % Then the delayed:
    [delayedSinograms, count] = fread(fid, structSizeSino.numTheta*structSizeSino.numR*numSinograms, readingFormat);
    % Close the file:
    fclose(fid);
    %Convert to single:
    sinograms = single(sinograms);
    delayedSinograms = single(delayedSinograms);
    % Rearrange in a 3d matrix:
    sinograms = reshape(sinograms, [structSizeSino.numR structSizeSino.numTheta numSinograms]);
    delayedSinograms = reshape(delayedSinograms, [structSizeSino.numR structSizeSino.numTheta numSinograms]);
else
    % Read raw data from uncomrpessed sinogram:
    [fid, message] = fopen(info.NameOfDataFile,'r');
    if fid == -1
        disp(ferror(fid));
    end
    % First the prompts:
    [sinograms, count] = fread(fid, structSizeSino.numTheta*structSizeSino.numR*numSinograms, readingFormat);
    % Then the delayed is empty:
    delayedSinograms = [];
    % Close the file:
    fclose(fid);
    %Convert to single:
    sinograms = single(sinograms);
    % Rearrange in a 3d matrix:
    sinograms = reshape(sinograms, [structSizeSino.numR structSizeSino.numTheta numSinograms]);
end

