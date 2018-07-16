% *********************************************************************
% Reconstruction Framework for Siemens Biograph mMR.
% class: PETDataClass
% Authors: Martin Belzunce, Abolfazl Mehranian. Kings College London.
% Date: 01/03/2018
% *********************************************************************
% Detects motion from list mmode
function [massCentroidAxial, medianAxial, modeAxial, massCentroidX_mm, massCentroidY_mm, massCentroidZ_mm, medianX_mm, medianY_mm, medianZ_mm, modeX_mm, ...
    modeY_mm, modeZ_mm, countRate] = detect_motion_from_list_mode(ObjData, window_sec)

if nargin == 1
    window_sec = 20;
end
% if multiple list-mode files, FrameTimePoints = [0, imageDuration(1); 0, imageDuration(2);...]
% if single list-mode files, FrameTimePoints = user defined frames or default [0,imageDuration(1)]
if strcmpi(ObjData.Data.Type, 'dicom_listmode')
    N = ObjData.Data.DCM.nListModes;
elseif strcmpi(ObjData.Data.Type, 'dicom_listmodelarge')
    N = ObjData.Data.DCM.nListModeLarges;
else
    N = ObjData.Data.IF.nListModeFiles;
end

% Size of mMr Sinogram's
numTheta = 252; numR = 344; numRings = 64; maxAbsRingDiff = 60; rFov_mm = 594/2; zFov_mm = 258; span = 1; widthRings_mm = 4.0571289;

structSizeSino3d = getSizeSino3dFromSpan(numR, numTheta, numRings, rFov_mm, zFov_mm, span, maxAbsRingDiff);
numDetectorsPerRing = 504;
numDetectors = numDetectorsPerRing*numRings;
[mapaDet1Ids, mapaDet2Ids] = createMmrDetectorsIdInSinogram3d(0);
% axial values:
ptrAxialvalues_mm = widthRings_mm/2 + widthRings_mm*[0:numRings-1];

fprintf('MATLAB listmode motion detection...., \n')
scanTime_sec = ObjData.FrameTimePoints(end);
numFrames = ceil(scanTime_sec/window_sec);
massCentroidAxial = zeros(numFrames,1);
massCentroidX_mm = zeros(numFrames,1);
massCentroidY_mm = zeros(numFrames,1);
massCentroidZ_mm = zeros(numFrames,1);
countRate = zeros(numFrames,1);
eventsPerFrame = zeros(numFrames,1);
sameFirstLastFrame = 0;
lastFrame = -1;
for j = 1: N
    % Read the list mode file.
    info = getInfoFromSiemensIntf(ObjData.Data.emission_listmode_hdr(j).n);
    binary_file = [ObjData.Data.emission_listmode(j).n]; % In the info sometimes is the wrong info
    % Open file:
    fid = fopen(binary_file, 'r');
    if fid == -1
        error('Error: List-mode binary file not found.');
    end
    % Read chunk of events: 1x10^6:
    seconds_per_mark = 0.001;
    event_size_bytes = 4;
    chunk_size_events = 10000000;
    chunk_size_bytes = chunk_size_events *event_size_bytes;

    % I fill one sinogram per frame:
    data = [];
    offsetTime = 0;
    flagOffsetSet = 0;
    % Read the list-mode:
    while(~feof(fid))
        % data need to be empty to read the file, if not keep
        % processing
        [data, count]=fread(fid,chunk_size_events,'*uint32');
        
        % calls matlab's accumarray to generate prompts/delays
        % sinograms from the provided lists
        % Process tags:
        % Timing events:
        tagTimeEventsMask = logical(bitshift(data,-29)==4);
        indicesTime = find(tagTimeEventsMask);
        % Time marker:
        elapsed_time_marker_msec = bitand(data(tagTimeEventsMask),hex2dec('1fffffff'));
        % Dead time marker with singles rate per block:
        tagDeadTimeTrackerMask = logical(bitshift(data,-29)==5);
        dead_time_marker_msec = bitand(data(tagTimeEventsMask),hex2dec('1fffffff'));

        % tagGantryEventsMask = logical(bitxor(bitshift(data,-29),1)==7);
        % tagPatientEventsMask = logical(bitxor(bitshift(data,-28),1)==15);
        % tagControlEventsMask = logical(bitxor(bitshift(data,-28),0)==15);

        % Get time stamps:
        timeStamps_sec = single(elapsed_time_marker_msec) .* seconds_per_mark;
        indexFrame = ceil((timeStamps_sec+1e-25)/window_sec);
        indexFrame(indexFrame>numFrames) = numFrames;
        firstFrame = indexFrame(1);
        % if the first frame of this chunk is the same as the last frame,
        % take it into account when computing the median:
        if firstFrame == lastFrame
            sameFirstLastFrame = 1;
        end
        lastFrame = indexFrame(end);
                
        for i = firstFrame : lastFrame
            indexThisFrame = find(indexFrame == i);
            % Now get the global indices from the time mark indices:
            indexPromptsFrame = indicesTime(indexThisFrame(1)):indicesTime(indexThisFrame(end));
            % Fill sinograms:
            dataThisFrame = data(indexPromptsFrame);
            P = ~bitor(bitshift(dataThisFrame,-31),0) & bitand(bitshift(dataThisFrame,-30),2^1-1);   % bit 31=0, bit 30=1 prompts, bit30=1 delays
            pba = double(bitand(dataThisFrame(P==1),2^30-1))+1; % prompts bin address, 2^30-1 is equal to hex2dec('3FFFFFFF')
            rba = double(bitand(dataThisFrame(P==0),2^30-1))+1; % randoms bin address, 2^30-1 is equal to hex2dec('3FFFFFFF')

            right_bits = (pba <= prod(ObjData.sinogram_size.matrixSize)) ;
            pba = pba(right_bits);

            right_bits = (rba <= prod(ObjData.sinogram_size.matrixSize)) ;
            rba = rba(right_bits);
            % Get z coordinate of trues:
            det1Ids = mapaDet1Ids(pba);
            det2Ids = mapaDet2Ids(pba);
            % get bin coordinates
            ring1 = floor((single(det1Ids)-1)/numDetectorsPerRing)+1;
            ring2 = floor((single(det2Ids)-1)/numDetectorsPerRing)+1;
            [indR, indTheta, indZ]=ind2sub(ObjData.sinogram_size.matrixSize,pba);
            [coordX1, coordY1, coordZ1, coordX2, coordY2, coordZ2, r, theta_deg] = getLorCoordinatesFromDet(indTheta',indR', ring1', ring2');
            massCentroidAxial(i) = massCentroidAxial(i) + sum((ring1+ring2)/2);
            massCentroidX_mm(i) = massCentroidX_mm(i) + sum((coordX1+coordX2)/2);
            massCentroidY_mm(i) = massCentroidY_mm(i) + sum((coordY1+coordY2)/2);
            massCentroidZ_mm(i) = massCentroidZ_mm(i) + sum((coordZ1+coordZ2)/2);

            if sameFirstLastFrame
                modeAxial(i) = mode([(ring1+ring2)/2; lastFrameAxialValues]);
                modeX_mm(i) = mode([(coordX1+coordX2)/2 lastFrameX_mm]);
                modeY_mm(i) = mode([(coordY1+coordY2)/2 lastFrameY_mm]);
                modeZ_mm(i) = mode([(coordZ1+coordZ2)/2 lastFrameZ_mm]);
                medianAxial(i) = median([(ring1+ring2)/2; lastFrameAxialValues]);
                medianX_mm(i) = median([(coordX1+coordX2)/2 lastFrameX_mm]);
                medianY_mm(i) = median([(coordY1+coordY2)/2 lastFrameY_mm]);
                medianZ_mm(i) = median([(coordZ1+coordZ2)/2 lastFrameZ_mm]);
            else
                modeAxial(i) = mode((ring1+ring2)/2);
                modeX_mm(i) = mode((coordX1+coordX2)/2);
                modeY_mm(i) = mode((coordY1+coordY2)/2);
                modeZ_mm(i) = mode((coordZ1+coordZ2)/2);
                medianAxial(i) = median((ring1+ring2)/2);
                medianX_mm(i) = median((coordX1+coordX2)/2);
                medianY_mm(i) = median((coordY1+coordY2)/2);
                medianZ_mm(i) = median((coordZ1+coordZ2)/2);
            end
            eventsPerFrame(i) = eventsPerFrame(i) + numel(pba);
        end
        % Save lastFrame events to be sued in the next read to compute the
        % median value:
        lastFrameAxialValues = (ring1+ring2)/2;
        lastFrameX_mm = (coordX1+coordX2)/2;
        lastFrameY_mm = (coordY1+coordY2)/2;
        lastFrameZ_mm = (coordZ1+coordZ2)/2;
    end
    massCentroidAxial = massCentroidAxial ./ eventsPerFrame;
    massCentroidX_mm = massCentroidX_mm ./ eventsPerFrame;
    massCentroidY_mm = massCentroidY_mm ./ eventsPerFrame;
    massCentroidZ_mm = massCentroidZ_mm ./ eventsPerFrame;
    countRate = eventsPerFrame./timeStamps_sec(end);
    fclose(fid);

end
