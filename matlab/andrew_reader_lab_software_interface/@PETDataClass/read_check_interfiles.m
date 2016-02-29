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
    tag = [0 0 0 0];
    for i = 1:numel(files)
        if ~isempty(strfind(files{i},'.n.hdr'))
            PETData.DataPath.norm = [FolderName '\' files{i}];
            tag(1)= 1;
        end
        if ~isempty(strfind(files{i},'.s.hdr')) && isempty(strfind(files{i},'uncomp.s.hdr'))
            PETData.DataPath.emission = [FolderName '\' files{i}];
            tag(2) = 1;
        end
        if ~isempty(strfind(files{i},'umap_hardware.mhdr'))
            PETData.DataPath.hardware_umap = [FolderName '\' files{i}];
            tag(3) = 1;
        end
        if ~isempty(strfind(files{i},'umap_human.mhdr')) || ~isempty(strfind(files{i},'umap.mhdr'))
            PETData.DataPath.umap = [FolderName '\' files{i}];
            tag(4) = 1;
        end
        if ~isempty(strfind(files{i},'uncomp.s.hdr'))
            PETData.DataPath.emission_uncomp = [FolderName '\' files{i}];
        end
    end
    if any(tag==0),
        msg = [];
        if tag(1)==0, msg = ['Normalization ''.n.hdr'' ' msg];end
        if tag(2)==0, msg = ['Emission sinogram ''.s.hdr'' ' msg];end
        if tag(3)==0, msg = [' ''umap_hardware.mhdr'' ' msg];end
        if tag(4)==0, msg = [' ''umap_human.mhdr'' ' msg];end
        error([ msg 'was not found in ' FolderName]);
    end
    PETData.DataPath.rawdata_sino = [FolderName '\rawdata_sino' ];
    PETData.DataPath.scatters = [FolderName '-00-scatter.mhdr'];     
end