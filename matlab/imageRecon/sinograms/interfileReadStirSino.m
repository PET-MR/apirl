%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 13/04/2015
%  *********************************************************************
% 
%
%  This funciton reads a sinogram in the interfile format for stir and
%  stores it in a 3d matrix with the size of the michelogram defined by the
%  structSize.In stir the sinograms are in fact viewgrams, so they must be
%  rearrenged to sinograms. In addition the segments are in different
%  order and they also use z2-z1 instead of z1-z2. So we switch negative
%  segments with postiive segments.
%  Example of this sinograms: 
%
%
% !INTERFILE  :=
% !imaging modality := PET
% name of data file :=/home/mab15/workspace/STIR/KCL/STIR_mMR_KCL/IM/EM_g1.s
% originating system := Siemens mMR
% !version of keys := STIR3.0
% !GENERAL DATA :=
% !GENERAL IMAGE DATA :=
% !type of data := PET
% imagedata byte order := LITTLEENDIAN
% !PET STUDY (General) :=
% !PET data type := Emission
% applied corrections := {None}
% !number format := float
% !number of bytes per pixel := 4
% number of dimensions := 4
% matrix axis label [4] := segment
% !matrix size [4] := 11
% matrix axis label [3] := view
% !matrix size [3] := 252
% matrix axis label [2] := axial coordinate
% !matrix size [2] := { 27,49,71,93,115,127,115,93,71,49,27}
% matrix axis label [1] := tangential coordinate
% !matrix size [1] := 344
% minimum ring difference per segment := { -60,-49,-38,-27,-16,-5,6,17,28,39,50}
% maximum ring difference per segment := { -50,-39,-28,-17,-6,5,16,27,38,49,60}
% Scanner parameters:= 
% Scanner type := Siemens mMR
% Number of rings                          := 64
% Number of detectors per ring             := 504
% Inner ring diameter (cm)                 := 65.6
% Average depth of interaction (cm)        := 0.7
% Distance between rings (cm)              := 0.40625
% Default bin size (cm)                    := 0.208626
% View offset (degrees)                    := 0
% Maximum number of non-arc-corrected bins := 344
% Default number of arc-corrected bins     := 344
% Number of blocks per bucket in transaxial direction         := 1
% Number of blocks per bucket in axial direction              := 2
% Number of crystals per block in axial direction             := 8
% Number of crystals per block in transaxial direction        := 9
% Number of detector layers                                   := 1
% Number of crystals per singles unit in axial direction      := 16
% Number of crystals per singles unit in transaxial direction := 9
% end scanner parameters:=
% effective central bin size (cm) := 0.208815
% number of time frames := 1
% image duration (sec)[1] := 1
% image relative start time (sec)[1] := 0
% !END OF INTERFILE :=



function [sinogram, structSizeSino] = interfileReadStirSino(filenameHeader)

% Read the whole header:
fid = fopen(filenameHeader);
if (fid == -1)
    ferror(fid);
end
textFields = textscan(fid, '%s%s', inf, 'Delimiter', {':=','\n'});
fclose(fid);

% Get the size of the sinogram. Temporarily we don't check the labels of
% each dimensions. By default, 
numR = getParameterValue(textFields, '!matrix size [1]');
numTheta = getParameterValue(textFields, '!matrix size [3]');
numSegments = getParameterValue(textFields, '!matrix size [4]');
% The sinograms per segment will be in a text string, process to convert it
% to an array:
sinogramsPerSegment_stir_text = getParameterValue(textFields, '!matrix size [2]');
sinogramsPerSegment_stir = textscan(sinogramsPerSegment_stir_text(2:end-1), '%d',inf, 'Delimiter', ',');
sinogramsPerSegment_stir = sinogramsPerSegment_stir{1};
% Minimum and maximum ring difference:
minRingDiff_stir_text = getParameterValue(textFields, 'minimum ring difference per segment');
minRingDiff_stir = textscan(minRingDiff_stir_text(2:end-1), '%d',inf, 'Delimiter', ',');
minRingDiff_stir = minRingDiff_stir{1};
maxRingDiff_stir_text = getParameterValue(textFields, 'maximum ring difference per segment');
maxRingDiff_stir = textscan(maxRingDiff_stir_text(2:end-1), '%d',inf, 'Delimiter', ',');
maxRingDiff_stir = maxRingDiff_stir{1};
% Number of rings:
numZ = getParameterValue(textFields, 'Number of rings');
% Maximum difference of rings:
maxAbsRingDiff = maxRingDiff_stir(end);
% radioScanner:
radioScanner_mm =  getParameterValue(textFields, 'Inner ring diameter (cm)')*10/2;
numDetectorsPerRing = getParameterValue(textFields, 'Number of detectors per ring');
% radioFov:
radioFov_mm = radioScanner_mm * sin(numR*pi/(numDetectorsPerRing*2));
% zFov:
distBetweenRing = getParameterValue(textFields, 'Distance between rings (cm)');
zFov_mm = distBetweenRing*numZ;


centerSegment = ceil(numel(sinogramsPerSegment_stir) / 2);
% The segments are odd. The segment in the middle is the direct singorams
% in stir:
sinogramsPerSegment(1) = sinogramsPerSegment_stir(centerSegment);
minRingDiff(1) = minRingDiff_stir(centerSegment);
maxRingDiff(1) = maxRingDiff_stir(centerSegment);
for i = 2 : 2: numel(sinogramsPerSegment_stir)
    % The positive diff first:
    sinogramsPerSegment(i) = sinogramsPerSegment_stir(centerSegment+round(i/2));
    minRingDiff(i) = minRingDiff_stir(centerSegment+round(i/2));
    maxRingDiff(i) = maxRingDiff_stir(centerSegment+round(i/2));
    % The negative diff:
    sinogramsPerSegment(i+1) = sinogramsPerSegment_stir(centerSegment-round(i/2));
    minRingDiff(i+1) = minRingDiff_stir(centerSegment-round(i/2));
    maxRingDiff(i+1) = maxRingDiff_stir(centerSegment-round(i/2));
end

% Create the struct:
structSizeSino = getSizeSino3dStruct(numR, numTheta, numZ, radioFov_mm, zFov_mm, sinogramsPerSegment, minRingDiff, maxRingDiff, maxAbsRingDiff);

% Name of the binary file:
filenameBinaryData = getParameterValue(textFields, 'name of data file');

% In stir the data is stored in viewgrams per segment, so it can't be puted
% in a 3d matrix. Because first it goes the first segment with a matrix of
% numRxnumSinosThisSegmenxnumTheta, and then with the next segment the
% second dimension changes. So I will process segment by segmente and
% write one at a time.
fid = fopen(filenameBinaryData, 'rb');
if(fid == -1)
    fprintf('Binary file %s couldn''t be opened for reading.', filenameBinaryData);
end

% Initialize sinogram:
sinogram = zeros(numR, numTheta, sum(sinogramsPerSegment));

% I have to write it in the order of the sinogram stir, so for each segment
% of stir I find the equivalent of the traditional sinogram.
for i = 1 : numel(sinogramsPerSegment_stir)
    % Read the first segment:
    sinograms_stir_thisSegment = fread(fid, [sinogramsPerSegment_stir(i)*numR*numTheta], 'single'); % Write in single.
    sinograms_stir_thisSegment = reshape(sinograms_stir_thisSegment, [numR sinogramsPerSegment_stir(i) numTheta]);
    sinograms_stir_thisSegment = permute(sinograms_stir_thisSegment, [1 3 2]);
    % Get the index of segments for the stir sinogram:
    % This would be the way if the segments in stir were the same as in
    % siemens and in apirl:
%     if(minRingDiff_stir(i) > 0) && (maxRingDiff_stir(i) > 0)
%         indexSegmentSino = 2*i - numel(sinogramsPerSegment_stir) - 1;
%     elseif(minRingDiff_stir(i) < 0) && (maxRingDiff_stir(i) < 0)
%         indexSegmentSino = numel(sinogramsPerSegment_stir) - 2*(i-1);
%     else
%         indexSegmentSino = 1;
%     end
    % But in stir is used z2-z1 instead of z1-z2, so we have to invert the order -1 the possitive:
    if(minRingDiff_stir(i) > 0) && (maxRingDiff_stir(i) > 0)
        indexSegmentSino = 2*i - numel(sinogramsPerSegment_stir);
    elseif(minRingDiff_stir(i) < 0) && (maxRingDiff_stir(i) < 0)
        indexSegmentSino = numel(sinogramsPerSegment_stir) - 2*(i-1) - 1;
    else
        indexSegmentSino = 1;
    end
    % Get the index of sinos for this segment counting the sinograms:
    indiceBaseSino = 0;
    for j = 1 : indexSegmentSino-1
        indiceBaseSino = indiceBaseSino + structSizeSino.sinogramsPerSegment(j);
    end
    indicesSino = (indiceBaseSino+1) : (indiceBaseSino+ structSizeSino.sinogramsPerSegment(indexSegmentSino));
    sinogram(:,:,indicesSino) = sinograms_stir_thisSegment;
end
fclose(fid);

% Text fields is a cell array where the first row is the field names, and
% the second one with its value:
function value = getParameterValue(textFields, strParameter)
    indexParameter = strncmp(strParameter, textFields{1}, numel(strParameter));
    [value,OK] = str2num(textFields{2}{indexParameter});
    if OK == 0
        value = textFields{2}{indexParameter};
    end
