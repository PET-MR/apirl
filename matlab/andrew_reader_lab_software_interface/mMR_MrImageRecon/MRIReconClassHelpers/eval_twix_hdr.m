function prot = eval_twix_hdr(filename)
% Author: Philipp Ehses (philipp.ehses@tuebingen.mpg.de), Feb/05/2015
    if ~exist('filename','var') || isempty(filename)
        info = 'Please select binary file to read';
        [fname,pathname]=uigetfile('*.dat',info); 
        if isempty(pathname)
            return
        end
        filename=[pathname fname];
    else
        if ischar(filename) 
            % assume that complete path is given
            if  ~strcmpi(filename(end-3:end),'.dat');
                filename=[filename '.dat'];   %% adds filetype ending to file
            end
        else
            % filename not a string, so assume that it is the MeasID
            measID   = filename;
            filelist = dir('*.dat');
            filesfound = 0;
            for k=1:numel(filelist)
                if regexp(filelist(k).name,['^meas_MID0*' num2str(measID) '.*\.dat'])==1
                    if filesfound == 0
                        filename = filelist(k).name;
                    end
                    filesfound = filesfound+1;
                end
            end
            if filesfound == 0
                error(['File with meas. id ' num2str(measID) ' not found.']);
            elseif filesfound > 1
                disp(['Multiple files with meas. id ' num2str(measID) ' found. Choosing first occurence.']);
            end
        end
    end
    
    fid = fopen(filename,'r','l','US-ASCII');
    
    firstInt  = fread(fid,1,'uint32');
    secondInt = fread(fid,1,'uint32');

    % lazy software version check (VB or VD?)
    if and(firstInt < 10000, secondInt <= 64)
        version = 'vd';
        disp('Software version: VD (!?)');

        % number of different scans in file stored in 2nd in
        NScans = secondInt;
        measID = fread(fid,1,'uint32');
        fileID = fread(fid,1,'uint32');
        % measOffset: points to beginning of header, usually at 10240 bytes
        measOffset = fread(fid,1,'uint64');
        measLength = fread(fid,1,'uint64');
        fseek(fid,measOffset,'bof');
        hdrLength  = fread(fid,1,'uint32');

    else
        % in VB versions, the first 4 bytes indicate the beginning of the
        % raw data part of the file
        version  = 'vb';
        disp('Software version: VB (!?)');
        measOffset = 0;
        hdrLength  = firstInt;
        NScans     = 1; % VB does not support multiple scans in one file
    end
    
    
    cPos = measOffset;
    prot = cell(1,NScans);
    for s=1:NScans
        fseek(fid,cPos,'bof');
        hdr_len = fread(fid, 1,'uint32');
        prot{s} = read_twix_hdr(fid);
%         cPos = cPos + ??; %wip multi-scan not working yet
    end
    fclose(fid);
    if NScans == 1
        prot = prot{1};
    end

    
end
    