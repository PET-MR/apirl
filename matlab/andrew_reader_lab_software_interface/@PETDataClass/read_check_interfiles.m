% *********************************************************************
% Reconstruction Framework for Siemens Biograph mMR.  
% class: PETRawData
% Authors: Martin Belzunce, Abolfazl Mehranian. Kings College London.
% Date: 22/02/2016
% *********************************************************************

function PETData = read_check_interfiles(PETData, FolderName)
    % todo: generate mhdr for PETData.DataPath to be consistant
    % todo: list mode data
    d = dir(FolderName);
    files = {d(3:end).name}';
    tag = [0 0 0 0 0];
    for i = 1:numel(files)
        if ~isempty(strfind(files{i},'.n.hdr'))
            PETData.DataPath.norm = [FolderName PETData.bar files{i}];
            tag(1)= 1;
        end
        if ~isempty(strfind(files{i},'.mhdr')) && isempty(strfind(files{i},'uncomp.s.hdr'))
            if isempty(strfind(files{i},'AC.mhdr')) && isempty(strfind(files{i},'human.mhdr')) && isempty(strfind(files{i},'hardware.mhdr'))
                PETData.DataPath.emission = [FolderName PETData.bar files{i}];
                tag(2) = 1;
            end
        end
        if ~isempty(strfind(files{i},'umap_hardware.mhdr'))
            PETData.DataPath.hardware_umap = [FolderName PETData.bar files{i}];
            tag(3) = 1;
        end
        if ~isempty(strfind(files{i},'umap_human.mhdr')) || ~isempty(strfind(files{i},'umap.mhdr'))
            PETData.DataPath.umap = [FolderName PETData.bar files{i}];
            tag(4) = 1;
        end
        if ~isempty(strfind(files{i},'uncomp.s.hdr'))
            PETData.DataPath.emission_uncomp = [FolderName PETData.bar files{i}];
        end
        % List- mode interfile.
        if ~isempty(strfind(files{i},'.l'))
            PETData.DataPath.emission_listmode = [FolderName PETData.bar files{i}(1:end-1) 'hdr']; % The header is .hdr
            tag(5) = 1;
        end
    end
    if any(tag==0),
        msg = [];
        if tag(1)==0, msg = ['Normalization ''.n.hdr'' ' msg];end
        if tag(2)==0, msg = ['Emission sinogram ''.s.hdr'' ' msg];end
        if tag(3)==0, msg = [' ''umap_hardware.mhdr'' ' msg];end
        if tag(4)==0, msg = [' ''umap_human.mhdr'' ' msg];end
        if tag(5)==0, msg = ['Emission list-mode ''.l.hdr'' ' msg];end
        warning([ msg 'was not found in ' FolderName]);
    end
    PETData.DataPath.rawdata_sino = [FolderName PETData.bar 'rawdata_sino' ];
    PETData.DataPath.scatters = [FolderName '-00-scatter.mhdr'];     
end