% *********************************************************************
% Reconstruction Framework for Siemens Biograph mMR.
% class: PETRawData
% Authors: Martin Belzunce, Abolfazl Mehranian. Kings College London.
% Date: 22/02/2016
% *********************************************************************

function PETData = prompt_JSRecon12(PETData, FolderName,reFraming)

if nargin==2
    reFraming = 0; % used for reframing listmode data without recalling JSRecon
end

[PathName,Name] = fileparts(FolderName);
Root_path = [PathName PETData.bar Name '-Converted' PETData.bar];

if ~exist(Root_path,'dir') && reFraming
    [status,~] = system([PETData.SoftwarePaths.e7.JSRecon12 ' ' FolderName]);
    if status
        error('JSRecon12 failed to generate sinograms');
    end
end

% get info from JSRecon12Info.txt in Root_path
fid = fopen([Root_path 'JSRecon12Info.txt'],'r');
w = fread(fid,'*char')';
fclose(fid);
LM = strfind(w,'LMFiles');

if ~isempty(LM)
    PETData.isListMode  = str2double(w(LM+[14:17])); % number of list-mode files found
    %         SD = strfind(w,'ImageDuration');
    %         PETData.ScanDuration_sec = str2double(w(SD+[18:23])); % search for lines instead of numbers
    PETData.NumberOfFrames = length(PETData.FrameTimePoints)-1;
%     if PETData.isListMode>1
%         error('%d list-mode files were found in %s, please seperate them into different folders\n',PETData.isListMode,FolderName)
%     end
else
    PETData.isListMode = 0;
end
SN = strfind(w,'SinoFiles');
if ~isempty(SN)
    PETData.isSinogram  = str2double(w(SN+[14:17])); % number of list-mode files found
else
    PETData.isSinogram = 0;
end




PETData.DataPath.norm  = [Root_path Name '-norm.n.hdr'];
if PETData.isSinogram
    for i = 1: PETData.isSinogram
        No = num2str(i-1,'%1.2d');
        PETData.DataPath.emission(i).n        = [Root_path Name '-' No PETData.bar Name '-' No '-sino.mhdr'];
        PETData.DataPath.umap(i).n            = [Root_path Name '-' No PETData.bar Name '-' No '-umap.mhdr'];
        PETData.DataPath.hardware_umap(i).n   = [Root_path Name '-' No PETData.bar Name '-' No '-umap-hardware.mhdr'];
        PETData.DataPath.rawdata_sino(i).n    = [Root_path Name '-' No PETData.bar 'rawdata_sino' ];
        PETData.DataPath.scatters(i).n        = [Root_path Name '-' No PETData.bar Name '-' No '-scatter.mhdr'];
        %             PETData.DataPath.emission_uncomp = []; % todo
    end
end


if PETData.isListMode
    i = 1;
    No = num2str(i-1,'%1.2d');
    
    PETData.DataPath.umap(i).n            = [Root_path Name '-LM-' No PETData.bar Name '-LM-' No '-umap.mhdr'];
    PETData.DataPath.hardware_umap(i).n   = [Root_path Name '-LM-' No PETData.bar Name '-LM-' No '-umap-hardware.mhdr'];
    
    PETData.DataPath.scatters(i).n        = [Root_path Name '-LM-' No PETData.bar Name '-LM-' No '-scatter.mhdr'];
    PETData.DataPath.lmhd                 = [Root_path Name '-LM-' No PETData.bar Name '-LM-' No '.hdr'];
    PETData.DataPath.lmdat                 = [Root_path Name '-LM-' No PETData.bar Name '-LM-' No '.l'];
    if reFraming
        PETData.DataPath.Name  =[Root_path Name '-LM-' No PETData.bar Name '-LM-' No '-sino-' num2str(reFraming)];
    else
        PETData.DataPath.Name  =[Root_path Name '-LM-' No PETData.bar Name '-LM-' No '-sino'];
    end
    %call Histogram_replay to generate the sinograms per frame
    fprintf('Histogram replay\n');
    PETData.histogram_data();
    
    for i = 1: PETData.NumberOfFrames
        if reFraming %avoid overwritting the data
            PETData.DataPath.emission(i).n        = [Root_path Name '-LM-' No PETData.bar Name '-LM-' No '-sino-' num2str(reFraming) '-' num2str(i-1) '.mhdr'];
            PETData.DataPath.rawdata_sino(i).n    = [Root_path Name '-LM-' No PETData.bar 'rawdata_sino-' num2str(reFraming) '-' num2str(i-1)];
        else
            PETData.DataPath.emission(i).n        = [Root_path Name '-LM-' No PETData.bar Name '-LM-' No '-sino-' num2str(i-1) '.mhdr'];
            PETData.DataPath.rawdata_sino(i).n    = [Root_path Name '-LM-' No PETData.bar 'rawdata_sino-' num2str(i-1)];
        end
        PETData.make_mhdr(PETData.DataPath.emission(i).n)
        
    end
end

