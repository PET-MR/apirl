%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 10/02/2015
%  *********************************************************************
%  function [info, structSizeSino]  = getInfoFromInterfile(filename)
% 
%  This function reads the header of an interfile sinograms and gets some
%  useful information from it. Such as the filename of the raw data, the
%  singles rates per bucket,..
%  It is based on the interfileinfo of matlab.
% Optionally it also returns an structure with the size of the sinogram.


function [info, structSizeSino] = getInfoFromInterfile(filename)

if(strcmp(computer(), 'GLNXA64'))
    os = 'linux';
    pathBar = '/';
elseif(strcmp(computer(), 'PCWIN') || strcmp(computer(), 'PCWIN64'))
    os = 'windows';
    pathBar = '\';
else
    disp('OS not compatible');
    return;
end

info = [];
% check header file extension
if (isempty(filename) || ~ischar(filename))
    error('getInfoFromSiemensIntf:filenameNotChar', ...
          'Filename must be a character array.')
end
[fpath,name,ext] = fileparts(filename);
if isempty(ext)
    filename = [filename '.hdr'];
end
% Como los interfile yo los manejo siempre con el h33 e i33 siempre en el
% mismo path, pero que se puedan leer y crear desde otro path. Cuando el
% nombre del interfile a leer tiene un path además del nombre, dicho path
% también debo agregarselo al "data file name" que figura en el h33:
relativePath = '';
barras = strfind(filename, pathBar);
if ~isempty(barras)
    relativePath = filename(1 : barras(end));
end

% open file for parsing
fid = fopen(filename);
if fid == -1
    error('Images:interfileinfo:invalidFilename', ...
          'Can not find header file %s.', filename);
end

% initialize variables
bad_chars = '!()[]/-_%:+*';    % Added %,:, siemens interfile sues it.
dates = {'DateOfKeys' 'ProgramDate' 'PatientDob' 'StudyDate'};
times = {'StudyTime' 'ImageStartTime'};
found_header = 0;
found_end = 0;
line_num = 0;

% parse through the file
while (true)
    line_txt = fgetl(fid);
    % stop if no more lines
    if (line_txt == -1)
        break;
    end
    
    % Strip out comments.  Interfile v3.3 spec, paragraph I.4.H: "Key-value
    % pairs may have comments appended to them by preceding the comment with
    % a semicolon <;>.  Conversion programs can ignore all characters
    % including and following a semicolon  <;> to the end of line code.
    % Where no key is stated, for example when an ASCII line starts with a
    % semicolon, the whole line is treated as comment.
    line_txt = regexprep(line_txt, ';.*$', '');
        
    if (sum(isspace(line_txt)) == length(line_txt))
        % Line is empty, skip to the next.
        continue;

    else
        line_num = line_num+1;
        % find index of separator and issue warning if not found
        sep_ind = strfind(line_txt, ':=');
        if (isempty(sep_ind))
            fclose(fid);
            % if no separator on first non-empty line, then not in INTERFILE format
            if isempty(info)
                error('Images:interfileinfo:invalidFile', ...
                      '%s is not a valid INTERFILE file.', filename);
                
            % if not on first non-empty line, then invalid expression
            else
                error('Images:interfileinfo:noSeparator', ...
                      'Invalid expression in line %s of %s.', num2str(line_num), filename);
            end
        
        else
            field_str_ind = 1;
            value_str_ind = sep_ind+2;
            field = '';
            
            % parse string to extract field
            while (true)
                [str, count, errmsg, nextindex] = sscanf(line_txt(field_str_ind:sep_ind-1), '%s', 1);
                % check for duplicate header
                if (strcmp(str, '!INTERFILE'))
                    if (found_header == 1)
                        fclose(fid);
                        error('Images:interfileinfo:duplicateHeader', ...
                              'Duplicate Interfile header in line %d of %s.', line_num, filename);
                        
                    else
                        found_header = 1;
                    end
                end
                
                % break if no field in rest of string
                if (count == 0)
                    break;
                end
                
                % concatenate strings to form field
                if (strcmp(str, 'ID'))
                    field = [field str];
                    
                else
                    str = lower(str);
                    i = 1;
                    
                    % remove illegal characters
                    while (i <= length(str))
                        k = strfind(bad_chars, str(i));
                        if (~isempty(k))
                            if (k >= 6)
                                str = [str(1:i-1) upper(str(i+1)) str(i+2:length(str))];

                            else
                                str = [str(1:i-1) str(i+1:length(str))];
                            end
                            % Added:
                            % If I take out one chrarachter I need to check
                            % that position again, if not when two non valid
                            % characters are found the second one is not detected:
                            i = i-1;
                        end

                        i = i+1;
                    end
                    if ~isempty(str) % Just in case a word is formed by all illegal characters
                        field = [field upper(str(1)) str(2:length(str))];
                    end
                end
                
                field_str_ind = field_str_ind+nextindex-1;
            end
            
            % remove extra spaces from beginning of value string
            for i = value_str_ind:length(line_txt)
                if (~isspace(line_txt(i)))
                    break;
                end
            end
            
            value = strcat(line_txt(i:length(line_txt)), '');
            if (strcmp(field, 'VersionOfKeys'))
                if (~strcmp(value, '3.3'))
                    fclose(fid);
                    err_id = 'Images:interfileinfo:unsupportedVersion';
                    err_msg = 'Unsupported version of keys detected.';
                    error(err_id, err_msg);
                end
            end
            
            if isempty(value)
                value = '';
            end
                
            x = str2double(value);
            if (~isnan(x)) % && (isempty(strfind(dates, field))) && (isempty(strfind(times, field))))
                value = x;
            end
            
            % close file if end-of-file marker encountered
            if (strcmp(field, 'EndOfInterfile'))
                found_end = 1;
                break;
                
            else
                % check for header
                if (found_header == 0)
                    fclose(fid);
                    err_id = 'Images:interfileinfo:noHeader';
                    err_msg = 'Interfile header not found.';
                    error(err_id, err_msg);

                % store field and value
                elseif (~strcmp(field, 'Interfile'))
                    if (isfield(info, field))
                        if (ischar(info.(field)))
                            info.(field) = {info.(field) value};
                            
                        elseif (iscell(info.(field)))
                            info.(field){length(info.(field))+1} = value;
                            
                        else
                            info.(field) = [info.(field) value];
                        end
                        
                    else
                        info.(field) = value;
                    end
                end
            end
        end
    end
end

% In the siemens interfile there is no enfofinterfile line.
% % check for end of file marker
% if (found_end == 0)
%     fclose(fid);
%     err_id = 'Images:interfileinfo:unexpectedEOF';
%     err_msg = 'Unexpected end of file.';
%     error(err_id, err_msg);
% end

% close file
fclose(fid);
% Add the relative path to the binary filename, if there is no path in the
% interfile:
barras = strfind(info.NameOfDataFile, pathBar);
if isempty(barras)
    info.NameOfDataFile = [relativePath info.NameOfDataFile];
end

% if ~isfield(info, 'SmsMiHeaderNameSpace')
%     structSizeSino = [];
%     return;
% end
    
if isfield(info, 'SmsMiHeaderNameSpace')
    % Siemens interfiles:
    % Post processing, some fields are processed to get them in a more useful
    % format. This changes if its an image or a sinogram
    if strcmp(info.SmsMiHeaderNameSpace, 'sinogram subheader')
        % Sinograms specific parameters:
        if (isfield(info, 'NumberOfBuckets'))
            numBuckets = info.NumberOfBuckets;
            singlesPerBucket = zeros(1, numBuckets);
        else
            perror('The number of bucket was not found.');
        end
        for i = 1 : numBuckets
            field = sprintf('BucketSinglesRate%d',i);
            if (isfield(info, field))
                singlesPerBucket(i) = info.(field);
            else
                perror(sprintf('The singles per bucket for the bucket number %d was not found.', i));
            end
            % After processing it I remove the field, because I will put just an
            % array:
            info = rmfield(info, field);
        end
        info.SinglesPerBucket = singlesPerBucket;

        % Process the segment table. Take out spaces and {}:
        charsToRemove = ['{','}',' '];
        for i = 1 : numel(charsToRemove)
            pos = find(info.SegmentTable == charsToRemove(i));
            info.SegmentTable(pos) = [];
        end
        % Separate by , and convert to number array
        cellStrNum = strsplit(info.SegmentTable,',');
        % Replace SegmentTable field for a numeric array:
        info.SegmentTable = zeros(1,numel(cellStrNum));
        if numel(cellStrNum) ~=  info.NumberOfSegments
            perror('The number of segments is different than the number of elements in the segment table.');
        end
        for i = 1 : numel(cellStrNum)
            info.SegmentTable(i) = str2num(cellStrNum{i});
        end

        % Get sino struct:
        rFov_mm = (info.MatrixSize1 * info.ScaleFactorMmPixel1) / 2;
        zFov_mm = (info.ScaleFactorMmPixel3 * (2*info.NumberOfRings+1));
        structSizeSino = getSizeSino3dFromSpan(info.MatrixSize1, info.MatrixSize2, info.NumberOfRings, rFov_mm, zFov_mm, info.AxialCompression, info.MaximumRingDifference);
    else
        % Siemens image:
        structSizeSino = [];
    end
elseif isfield(info, 'MinimumRingDifferencePerSegment') % Apirl sinogram
    % Process the segment table. Take out spaces and {}:
    charsToRemove = ['{','}',' '];
    for i = 1 : numel(charsToRemove)
        pos = find(info.MatrixSize3 == charsToRemove(i));
        info.MatrixSize3(pos) = [];
        
        pos = find(info.MinimumRingDifferencePerSegment == charsToRemove(i));
        info.MinimumRingDifferencePerSegment(pos) = [];
        
        pos = find(info.MaximumRingDifferencePerSegment == charsToRemove(i));
        info.MaximumRingDifferencePerSegment(pos) = [];
    end
    % Separate by , and convert to number array
    cellStrNum = strsplit(info.MatrixSize3,',');
    % Replace SegmentTable field for a numeric array:
    info.MatrixSize3 = zeros(1,numel(cellStrNum));
    for i = 1 : numel(cellStrNum)
        info.MatrixSize3(i) = str2num(cellStrNum{i});
    end
    info.NumberOfSegments = numel(info.MatrixSize3);
    
    % Separate by , and convert to number array
    cellStrNum = strsplit(info.MinimumRingDifferencePerSegment,',');
    if numel(cellStrNum) ~=  info.NumberOfSegments
        perror('The number of MinimumRingDifferencePerSegment is different than the number of segments in MatrixSize3.');
    end
    % Replace SegmentTable field for a numeric array:
    info.MinimumRingDifferencePerSegment = zeros(1,numel(cellStrNum));
    for i = 1 : numel(cellStrNum)
        info.MinimumRingDifferencePerSegment(i) = str2num(cellStrNum{i});
    end
    
    % Separate by , and convert to number array
    cellStrNum = strsplit(info.MaximumRingDifferencePerSegment,',');
    if numel(cellStrNum) ~=  info.NumberOfSegments
        perror('The number of MaximumRingDifferencePerSegment is different than the number of segments in MatrixSize3.');
    end
    % Replace SegmentTable field for a numeric array:
    info.MaximumRingDifferencePerSegment = zeros(1,numel(cellStrNum));
    for i = 1 : numel(cellStrNum)
        info.MaximumRingDifferencePerSegment(i) = str2num(cellStrNum{i});
    end
    
    % Return the size of struct size sino, with only the avialable info:
    structSizeSino = getSizeSino3dStruct(info.MatrixSize1, info.MatrixSize2, info.NumberOfRings, 0, 0, info.MatrixSize3, info.MinimumRingDifferencePerSegment, info.MaximumRingDifferencePerSegment, abs(info.MinimumRingDifferencePerSegment(end)));
elseif isfield(info, 'NumberOfProjections') % Apirl 2d sinogram
    structSizeSino = getSizeSino2dStruct(info.MatrixSize1, info.MatrixSize2, info.NumberOfProjections, 0, 0);
%elseif isfield(info, 'NumberOfImagesWindow')
%    structSizeSino = getSizeSino2dStruct(info.MatrixSize1, info.MatrixSize2, info.NumberOfImagesWindow, 0, 0);
else
    % Is an image:
    if isfield(info, 'MatrixSize3')&& isfield(info, 'ScalingFactorMmPixel3')
        % 3d image:
        structSizeSino = imref3d([info.MatrixSize2 info.MatrixSize1 info.MatrixSize3], info.ScalingFactorMmPixel2, info.ScalingFactorMmPixel1, info.ScalingFactorMmPixel3);
    else
        structSizeSino = imref2d([info.MatrixSize2 info.MatrixSize1], info.ScalingFactorMmPixel2, info.ScalingFactorMmPixel1);
    end
end