% *********************************************************************
% Reconstruction Framework for Siemens Biograph mMR.
% class: PETDataClass
% Authors: Martin Belzunce, Abolfazl Mehranian. Kings College London.
% Date: 01/03/2016
% *********************************************************************
% Histograms data for a given list mode. The frames duration is obtained
% from the PETData config.
% To do: process gantry position events and patien tracking evnets.
function data = histogram_data(PETData)

if strcmpi(PETData.MethodListData,'e7')
    fprintf('calling e7 histogram replay...\n')
    frame = [num2str(PETData.FrameTimePoints(1)) ':' ];
    for i = 2:length(PETData.FrameTimePoints)-1
        frame = [frame num2str(PETData.FrameTimePoints(i)) ','];
    end
    frame = [frame num2str(PETData.FrameTimePoints(end))];
    command = [PETData.SoftwarePaths.e7.HistogramReplay ' --lmhd "' PETData.DataPath.lmhd '"' ...
        ' --lmdat "' PETData.DataPath.lmdat '" --lmode PROMPTS_RANDOMS --opre ' PETData.DataPath.Name ' --frame ' frame];
    
    [status,message] = system(command);
    if status
        display(message)
        error('HistogramReplay was failed');
    end
elseif strcmpi(PETData.MethodListData,'matlab')
    fprintf('calling MATLAB histogrammer...\n')
    
    % Read the list mode file.
    info = getInfoFromSiemensIntf(PETData.DataPath.lmhd);
    binary_file = [PETData.DataPath.lmdat]; % In the info sometimes is the wrong info
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
    for i = 1 : PETData.NumberOfFrames
        % Create a sinogram:
        sino_prompts = zeros(prod(PETData.sinogram_size.matrixSize),1, 'single');
        sino_delays = zeros(prod(PETData.sinogram_size.matrixSize),1, 'single');
        
        % I have noticed that the events has an offset in time, so now i
        % take the first time from the data:
        startingTimeForThisFrame_sec = offsetTime + PETData.DynamicFrames_sec(i);
        endingTimeForThisFrame_sec = offsetTime + PETData.DynamicFrames_sec(i+1);
        flag_new_frame = 0;
        % Read the list-mode:
        while(~feof(fid)& ~flag_new_frame)
            % data need to be empty to read the file, if not keep
            % processing
            if isempty(data)
                [data, count]=fread(fid,chunk_size_events,'*uint32');
            end
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
            % Check for any offset in time
            if flagOffsetSet == 0
                offsetTime = timeStamps_sec(1)-seconds_per_mark;
                startingTimeForThisFrame_sec = offsetTime + PETData.DynamicFrames_sec(i);
                endingTimeForThisFrame_sec = offsetTime + PETData.DynamicFrames_sec(i+1);
                flagOffsetSet = 1;
            end
            % We don't need to check the intial time, because we are
            % removing previous frames files:
            indiceTimeStampsOutOfFrame = find(timeStamps_sec > endingTimeForThisFrame_sec);
            % if no timestamp out of frame, process thw whole data:
            if(sum(indiceTimeStampsOutOfFrame) ~= 0)
                indiceLimitFrame = indicesTime(indiceTimeStampsOutOfFrame(1));
                % Discard:
                dataFrame = data(1:indiceLimitFrame); % It could be indiceLimitFrame-1 because is not perfect the rpecision.
                % Remove old data:
                data(1:indiceLimitFrame) = [];
            else
                dataFrame = data;
                data = [];
            end
            % Fill sinograms:
            P = ~bitor(bitshift(dataFrame,-31),0) & bitand(bitshift(dataFrame,-30),2^1-1);   % bit 31=0, bit 30=1 prompts, bit30=1 delays
            pba = double(bitand(dataFrame(P==1),2^30-1))+1; % prompts bin address, 2^30-1 is equal to hex2dec('3FFFFFFF')
            rba = double(bitand(dataFrame(P==0),2^30-1))+1; % randoms bin address, 2^30-1 is equal to hex2dec('3FFFFFFF')
            
            right_bits = (pba <= prod(PETData.sinogram_size.matrixSize)) ;
            pba = pba(right_bits);
            
            right_bits = (rba <= prod(PETData.sinogram_size.matrixSize)) ;
            rba = rba(right_bits);
            
            sino_prompts = sino_prompts + accumarray(pba,1,[prod(PETData.sinogram_size.matrixSize),1]);
            sino_delays = sino_delays + accumarray(rba,1,[prod(PETData.sinogram_size.matrixSize),1]);
            disp(sprintf('Frame %d. %d new coincidences. Total counts: %d', i, numel(pba), sum(sino_prompts)));
            disp(sprintf('Frame %d. %d new delayed coincidences. Total delayed counts: %d', i, numel(rba), sum(sino_delays)));
            % if not empty new frame:
            if ~isempty(data)
                % Write both sinograms:
                sino_prompts = reshape(sino_prompts,PETData.sinogram_size.matrixSize);
                sino_delays = reshape(sino_delays,PETData.sinogram_size.matrixSize);
                interfileWriteSino(single(sino_prompts), [PETData.DataPath.path 'sinogram_frame_' num2str(i)], getSizeSino3dFromSpan(PETData.sinogram_size.nRadialBins, PETData.sinogram_size.nAnglesBins, PETData.sinogram_size.nRings, ...
                    296, 256, PETData.sinogram_size.span, PETData.sinogram_size.maxRingDifference));
                interfileWriteSino(single(sino_prompts), [PETData.DataPath.path 'sinogram_frame_' num2str(i) '_delayed'], getSizeSino3dFromSpan(PETData.sinogram_size.nRadialBins, PETData.sinogram_size.nAnglesBins, PETData.sinogram_size.nRings, ...
                    296, 256, PETData.sinogram_size.span, PETData.sinogram_size.maxRingDifference));
                flag_new_frame = 1;
            end
            
            
            
        end
    end
    % Write last frame:
    % Write both sinograms:
    sino_prompts = reshape(sino_prompts,PETData.sinogram_size.matrixSize);
    sino_delays = reshape(sino_delays,PETData.sinogram_size.matrixSize);
    interfileWriteSino(single(sino_prompts), [PETData.DataPath.path 'sinogram_frame_' num2str(i)], getSizeSino3dFromSpan(PETData.sinogram_size.nRadialBins, PETData.sinogram_size.nAnglesBins, PETData.sinogram_size.nRings, ...
        296, 256, PETData.sinogram_size.span, PETData.sinogram_size.maxRingDifference));
    interfileWriteSino(single(sino_delays), [PETData.DataPath.path 'sinogram_frame_' num2str(i) '_delayed'], getSizeSino3dFromSpan(PETData.sinogram_size.nRadialBins, PETData.sinogram_size.nAnglesBins, PETData.sinogram_size.nRings, ...
        296, 256, PETData.sinogram_size.span, PETData.sinogram_size.maxRingDifference));
    fclose(fid);
    
    
end