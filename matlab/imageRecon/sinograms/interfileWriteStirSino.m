%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 25/02/2015
%  *********************************************************************
% 
%
%  This funciton writes a sinogram in the interfile format for stir.
%  Its fixed for Siemens Mmr sinograms. It changes the order of how its
%  stored to be compatible with stir. Its centered in the ring diff.
%  IMPORTANT: in stir the sinograms are in different order, because they
%  use z2-z1 instead of z1-z2. So we switch negative segments with postiive
%  segments.
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

% Only header permits to write only the header. Its useful to generate
% templates.

function interfileWriteStirSino(sinogram, filename, structSizeSino, onlyHeader)

% Only use the header only when its specified.
if nargin == 3
    onlyHeader = 0;
end

% Add extensions to both files:
filenameHeader = sprintf('%s.hs', filename);
filenameSino = sprintf('%s.s', filename);

% 3D sinogram:
if isfield(structSizeSino, 'sinogramsPerSegment')
    centerSegment = ceil(numel(structSizeSino.sinogramsPerSegment) / 2);
    % The segments are odd. The segment in the middle is the direct singorams
    % in stir:
    sinogramsPerSegment_stir(centerSegment) = structSizeSino.sinogramsPerSegment(1);
    minRingDiff_stir(centerSegment) = structSizeSino.minRingDiff(1);
    maxRingDiff_stir(centerSegment) = structSizeSino.maxRingDiff(1);
    for i = 2 : 2: numel(structSizeSino.sinogramsPerSegment)
        % The positive diff first:
        sinogramsPerSegment_stir(centerSegment+round(i/2)) = structSizeSino.sinogramsPerSegment(i);
        minRingDiff_stir(centerSegment+round(i/2)) = structSizeSino.minRingDiff(i);
        maxRingDiff_stir(centerSegment+round(i/2)) = structSizeSino.maxRingDiff(i);
        % The negative diff:
        sinogramsPerSegment_stir(centerSegment-round(i/2)) = structSizeSino.sinogramsPerSegment(i+1);
        minRingDiff_stir(centerSegment-round(i/2)) = structSizeSino.minRingDiff(i+1);
        maxRingDiff_stir(centerSegment-round(i/2)) = structSizeSino.maxRingDiff(i+1);
    end
else
    % sinogram 2d:
    sinogramsPerSegment_stir = 1;
    minRingDiff_stir = 0;
    maxRingDiff_stir = 0;
end
% Open header
fid = fopen(filenameHeader, 'w');
if(fid == -1)
    fprintf('File %s couldn''t be created.', filenameHeader);
end

% Writing each field:
fprintf(fid,'!INTERFILE :=\n');
fprintf(fid,'!imaging modality := PET \n');
% Name of data file. I do it without any absolute path:
barras = strfind(filenameSino, '/');
if ~isempty(barras)
    filenameSinoForHeader = filenameSino(barras(end)+1 : end);
else
    filenameSinoForHeader = filenameSino;
end
fprintf(fid,'name of data file := %s\n', filenameSinoForHeader);
fprintf(fid,'originating system := Siemens mMR\n');
fprintf(fid,'!version of keys := STIR3.0\n');
fprintf(fid,'!GENERAL DATA := \n');
fprintf(fid,'!GENERAL IMAGE DATA :=\n');
fprintf(fid,'!type of data := PET\n');
fprintf(fid,'imagedata byte order := LITTLEENDIAN\n');
fprintf(fid,'!PET STUDY (General) :=\n');
fprintf(fid,'!PET data type := Emission\n');
fprintf(fid,'applied corrections := {None}\n');
% Force to use float (casting of the sinogram  data later)
fprintf(fid,'!number format := float\n');
fprintf(fid,'!number of bytes per pixel := 4\n');
fprintf(fid,'number of dimensions := 4\n');
fprintf(fid,'matrix axis label [4] := segment\n');
fprintf(fid,'!matrix size [4] := %d\n', numel(sinogramsPerSegment_stir));
% Stir it saves views instead of sinograms:
fprintf(fid,'matrix axis label [3] := view\n');
fprintf(fid,'!matrix size [3] := %d\n', structSizeSino.numTheta);
fprintf(fid,'matrix axis label [2] := axial coordinate\n');
fprintf(fid,'!matrix size [2] := {');
fprintf(fid,' %d', sinogramsPerSegment_stir(1));
for i = 2 : numel(sinogramsPerSegment_stir)
     fprintf(fid,', %d', sinogramsPerSegment_stir(i));
end
fprintf(fid,' }\n');
fprintf(fid,'matrix axis label [1] := tangential coordinate\n');
fprintf(fid,'!matrix size [1] := %d\n', structSizeSino.numR);
fprintf(fid,'minimum ring difference per segment := { ');
fprintf(fid,' %d', minRingDiff_stir(1));
for i = 2 : numel(minRingDiff_stir)
     fprintf(fid,', %d', minRingDiff_stir(i));
end
fprintf(fid,' }\n');
fprintf(fid,'maximum ring difference per segment := { ');
fprintf(fid,' %d', maxRingDiff_stir(1));
for i = 2 : numel(maxRingDiff_stir)
     fprintf(fid,', %d', maxRingDiff_stir(i));
end
fprintf(fid,' }\n');

if isfield(structSizeSino, 'sinogramsPerSegment')
    % Fixed parameters for Siemens Mmr scanner:
    fprintf(fid,'Scanner parameters:= \n');
    fprintf(fid,'Scanner type := Siemens mMR\n');
    fprintf(fid,'Number of rings                          := 64\n');
    fprintf(fid,'Number of detectors per ring             := 504\n');
    fprintf(fid,'Inner ring diameter (cm)                 := 65.6\n');
    fprintf(fid,'Average depth of interaction (cm)        := 0.7\n');
    fprintf(fid,'Distance between rings (cm)              := 0.40625\n');
    fprintf(fid,'Default bin size (cm)                    := 0.208626\n');
    fprintf(fid,'View offset (degrees)                    := 0\n');
    fprintf(fid,'Maximum number of non-arc-corrected bins := 344\n');
    fprintf(fid,'Default number of arc-corrected bins     := 344\n');
    fprintf(fid,'Number of blocks per bucket in transaxial direction         := 1\n');
    fprintf(fid,'Number of blocks per bucket in axial direction              := 2\n');
    fprintf(fid,'Number of crystals per block in axial direction             := 8\n');
    fprintf(fid,'Number of crystals per block in transaxial direction        := 9\n');
    fprintf(fid,'Number of detector layers                                   := 1\n');
    fprintf(fid,'Number of crystals per singles unit in axial direction      := 16\n');
    fprintf(fid,'Number of crystals per singles unit in transaxial direction := 9\n');
    fprintf(fid,'end scanner parameters:=\n');
    fprintf(fid,'effective central bin size (cm) := 0.208815\n');
    fprintf(fid,'number of time frames := 1\n');
    fprintf(fid,'image duration (sec)[1] := 1\n');
    fprintf(fid,'image relative start time (sec)[1] := 0\n');
    fprintf(fid,'!END OF INTERFILE :=\n');
else
    % Fixed parameters for Siemens Mmr scanner:
    fprintf(fid,'Scanner parameters:= \n');
    fprintf(fid,'Scanner type := Siemens mMR\n');
    fprintf(fid,'Number of rings                          := 1\n');
    fprintf(fid,'Number of detectors per ring             := 504\n');
    fprintf(fid,'Inner ring diameter (cm)                 := 65.6\n');
    fprintf(fid,'Average depth of interaction (cm)        := 0.7\n');
    fprintf(fid,'Distance between rings (cm)              := 0.40625\n');
    fprintf(fid,'Default bin size (cm)                    := 0.208626\n');
    fprintf(fid,'View offset (degrees)                    := 0\n');
    fprintf(fid,'Maximum number of non-arc-corrected bins := 344\n');
    fprintf(fid,'Default number of arc-corrected bins     := 344\n');
    fprintf(fid,'Number of blocks per bucket in transaxial direction         := 1\n');
    fprintf(fid,'Number of blocks per bucket in axial direction              := 1\n');
    fprintf(fid,'Number of crystals per block in axial direction             := 1\n');
    fprintf(fid,'Number of crystals per block in transaxial direction        := 9\n');
    fprintf(fid,'Number of detector layers                                   := 1\n');
    fprintf(fid,'Number of crystals per singles unit in axial direction      := 1\n');
    fprintf(fid,'Number of crystals per singles unit in transaxial direction := 9\n');
    fprintf(fid,'end scanner parameters:=\n');
    fprintf(fid,'effective central bin size (cm) := 0.208815\n');
    fprintf(fid,'number of time frames := 1\n');
    fprintf(fid,'image duration (sec)[1] := 1\n');
    fprintf(fid,'image relative start time (sec)[1] := 0\n');
    fprintf(fid,'!END OF INTERFILE :=\n');
end
% Header ready:
fclose(fid);


% In stir the data is stored in viewgrams per segment, so it can't be puted
% in a 3d matrix. Because first it goes the first segment with a matrix of
% numRxnumSinosThisSegmenxnumTheta, and then with the next segment the
% second dimension changes. So I will process segment by segmente and
% write one at a time.
fid = fopen(filenameSino, 'wb');
if(fid == -1)
    fprintf('File %s couldn''t be created.', filenameImage);
end

if onlyHeader == 0
    % 3D sinogram:
    if isfield(structSizeSino, 'sinogramsPerSegment')
        % I have to write it in the order of the sinogram stir, so for each segment
        % of stir I find the equivalent of the traditional sinogram.
        for i = 1 : numel(sinogramsPerSegment_stir)
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
            sinogram_stir_thisSegment = zeros([structSizeSino.numR structSizeSino.numTheta sinogramsPerSegment_stir(i)], 'single');
            % Get the index of sinos for this segment counting the sinograms:
            indiceBaseSino = 0;
            for j = 1 : indexSegmentSino-1
                indiceBaseSino = indiceBaseSino + structSizeSino.sinogramsPerSegment(j);
            end
            indicesSino = (indiceBaseSino+1) : (indiceBaseSino+ structSizeSino.sinogramsPerSegment(indexSegmentSino));
            sinogram_stir_thisSegment = sinogram(:,:,indicesSino);
            % Interchange 2nd and 3rd dimensions to go from sinograms to viewgrams:
            sinogram_stir_thisSegment = permute(sinogram_stir_thisSegment, [1 3 2]);
            % Write to a file:
            fwrite(fid, sinogram_stir_thisSegment, 'single'); % Write in single.
        end
    else
        % 2d:
        fwrite(fid, sinogram, 'single'); % Write in single.
    end
end
fclose(fid);
