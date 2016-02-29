% *********************************************************************
% Reconstruction Framework for Siemens Biograph mMR.  
% class: PETRawData
% Authors: Martin Belzunce, Abolfazl Mehranian. Kings College London.
% Date: 22/02/2016
% *********************************************************************

function PETData = prompt_JSRecon12(PETData, FolderName)
            
    [PathName,Name] = fileparts(FolderName);
    Root_path = [PathName '\' Name '-Converted\'];

    if ~exist(Root_path,'dir')
        [status,~] = system([PETData.SoftwarePathes.e7.JSRecon12 FolderName]);
        if status
            error('JSRecon12 failed to generate sinograms');
        end
    end
    PETData.DataPath.emission        = [Root_path Name '-00\' Name '-00-sino.mhdr'];
    PETData.DataPath.norm            = [Root_path Name '-norm.n.hdr'];
    PETData.DataPath.umap            = [Root_path Name '-00\' Name '-00-umap.mhdr'];
    PETData.DataPath.hardware_umap   = [Root_path Name '-00\' Name '-00-umap-hardware.mhdr'];
    PETData.DataPath.rawdata_sino    = [Root_path Name '-00\rawdata_sino' ];
    PETData.DataPath.scatters        = [Root_path Name '-00\' Name '-00-scatter.mhdr'];
    PETData.DataPath.emission_uncomp = []; % todo
end