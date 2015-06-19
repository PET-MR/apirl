%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 03/02/2015
%  *********************************************************************
%  Replaces in a header the name of data file for a new one. It can be used
%  to add a  compelte path of the filename, replacing a relative filename.



function replaceDataFilenameInInterfile(filenameHeader, newDataFilename)

% Modify header file with new filename for main file
fid = fopen(filenameHeader, 'r');
if fid == -1
    error('%s not found!\n', filenameHeader);
else
    % Read in file
    file_strg = fscanf(fid, '%c');
    fclose(fid);
    
    % Find name of data file := and end of filename
    begin_idx = -1;
    if ~isempty(strfind(file_strg, 'name of data file :='))
        begin_idx = strfind(file_strg, 'name of data file :=') + length('name of data file :=');
    elseif ~isempty(strfind(file_strg, 'name of data file:='))
        begin_idx = strfind(file_strg, 'name of data file:=') + length('name of data file:=');
    else
        error('Neither "name of data file :=" nor "name of data file:=" found in header file.')
    end
    if length(begin_idx) > 1 % Check which one is not commented out
        warning('Multiple entries for "name of data file :=" or "name of data file:=" found.');
    end
    idx(1) = begin_idx(1);
    identf_idx = strfind(file_strg(idx(1):end), char(10));
    idx(2) = identf_idx(1) + idx(1) - 1;
    
    % Replace filename
    file_strg = [file_strg(1:idx(1)-1) newDataFilename file_strg(idx(2):end)];
    
    % Save updated header
    fid=fopen(filenameHeader,'w+'); 
    fprintf(fid,'%c',file_strg ); 
    fclose(fid);
end