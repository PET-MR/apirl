% *********************************************************************
% Reconstruction Framework for Siemens Biograph mMR.
% class: PETRawData
% Authors: Martin Belzunce, Abolfazl Mehranian. Kings College London.
% Date: 22/02/2016
% *********************************************************************

function PETData = read_histogram_interfiles(PETData, FolderName, reFraming)


if nargin ==2
    reFraming = 0;
end

isThis = @(inputFile,Type) ~isempty(strfind(inputFile,Type));
d = dir(FolderName);
file = {d(3:end).name}';
js = 0;
jl = 0;
for i = 1:numel(file)
    if isThis(file{i},'.n.hdr')
        PETData.DataPath.norm = [FolderName PETData.bar file{i}];
        tag(1)= 1;
    end
    
    % Emission sinograms
    if (isThis(file{i},'s.mhdr') || (isThis(file{i},'sino') && isThis(file{i},'.mhdr')) ) && ~isThis(file{i},'uncomp.s.mhdr')
    js = js +1;
        PETData.DataPath.emission(js).n = [FolderName PETData.bar file{i}];
    end
    % Attenuation maps
    if isThis(file{i},'umap_hardware.mhdr') || isThis(file{i},'umap-hardware.mhdr')
        PETData.DataPath.hardware_umap(1).n = [FolderName PETData.bar file{i}];
        tag(2) = 1;
    end
    if isThis(file{i},'umap_human.mhdr') || isThis(file{i},'umap.mhdr')
        PETData.DataPath.umap(1).n = [FolderName PETData.bar file{i}];
        tag(3) = 1;
    end
    
    if isThis(file{i},'uncomp.s.mhdr')
        PETData.DataPath.emission_uncomp = [FolderName PETData.bar file{i}];
    end
    % List- mode interfile.
    if isThis(file{i},'.l')
        jl = jl +1;
        PETData.DataPath.lmhd = [FolderName PETData.bar file{i}(1:end-1) 'hdr']; % The header is .hdr
        PETData.DataPath.lmdat =[FolderName PETData.bar file{i}];
        PETData.isListMode = 1;
        if jl>1
            error('There are more than one list-mode file, please provide one of them at a time\n');
        end
    end
end
PETData.isSinogram = js;
PETData.isListMode = jl;


if ~js && ~jl
    error('neither sinograms nor list-mode data were found\n');
end
if any(tag==0),
    msg = [];
    if tag(1)==0, msg = ['Normalization ''.n.hdr'' ' msg];end
    if tag(2)==0, msg = [' ''umap_hardware.mhdr'' ' msg];end
    if tag(3)==0, msg = [' ''umap_human.mhdr'' ' msg];end
    error([ msg 'was not found in ' FolderName]);
end

PETData.DataPath.scatters(1).n = [FolderName '-00-scatter.mhdr'];
if PETData.isSinogram
    for i = 1: PETData.isSinogram
        No = num2str(i-1,'%1.2d');
        PETData.DataPath.rawdata_sino(i).n = [FolderName PETData.bar 'rawdata_sino_' No ];
    end
end

if PETData.isListMode
    [Dir,Name]=fileparts(PETData.DataPath.lmdat) ;
    if reFraming
        PETData.DataPath.Name  =[Dir PETData.bar Name '-sino-' num2str(reFraming) ];
    else
        PETData.DataPath.Name  =[Dir PETData.bar Name '-sino'];
    end
    
        %call Histogram_replay to generate the sinograms per frame
    fprintf('Histogram replay\n');
    PETData.histogram_data();
    
    for i = 1: PETData.NumberOfFrames
        PETData.DataPath.emission(i).n        = [PETData.DataPath.Name  '-'  num2str(i-1) '.mhdr'];
        if reFraming %avoid overwritting the data
            PETData.DataPath.rawdata_sino(i).n    = [Dir  PETData.bar 'rawdata_sino-' num2str(reFraming) '-' num2str(i-1)];
        else
            PETData.DataPath.rawdata_sino(i).n    = [Dir  PETData.bar 'rawdata_sino-' num2str(i-1)];
        end
        PETData.make_mhdr(PETData.DataPath.emission(i).n)
        
    end
end

end