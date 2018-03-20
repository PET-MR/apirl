% *********************************************************************
% Reconstruction Framework for Siemens Biograph mMR.
% class: PETRawData
% Authors: Martin Belzunce, Abolfazl Mehranian. Kings College London.
% Date: 22/02/2016
% *********************************************************************

function ObjData = prompt_JSRecon12(ObjData, FolderName)



if ObjData.Data.DCM.nNormFiles >1
    error('%d normalization files were found in %s\n',ObjData.Data.DCM.nNormFiles,FolderName);
end


[PathName,Name] = fileparts(FolderName(1:end-1)); % remove the bar from address to get the folder's name

% Converted path:
Root_path = [PathName ObjData.bar Name '-Converted' ObjData.bar];

if ~exist(Root_path,'dir')
    [status,~] = system([ObjData.SoftwarePaths.e7.JSRecon12 ' ' FolderName]);
    if status
        error('JSRecon12 failed to generate sinograms');
    end
else
    fprintf('The directorty %s already exists\n',Root_path);
end
ObjData.Data.norm  = [Root_path Name '-norm.n.hdr'];

if strcmpi(ObjData.Data.Type, 'dicom_sinogram')
    for i = 1: ObjData.Data.DCM.nSinograms
        No = num2str(i-1,'%1.2d');
        ObjData.Data.emission(i).n        = [Root_path Name '-' No ObjData.bar Name '-' No '-sino.mhdr'];
        ObjData.Data.umap(i).n            = [Root_path Name '-' No ObjData.bar Name '-' No '-umap.mhdr'];
        ObjData.Data.hardware_umap(i).n   = [Root_path Name '-' No ObjData.bar Name '-' No '-umap-hardware.mhdr'];
        ObjData.Data.rawdata_sino(i).n    = [Root_path Name '-' No ObjData.bar 'rawdata_sino' ];
        ObjData.Data.scatters(i).n        = [Root_path Name '-' No ObjData.bar Name '-' No '-scatter.mhdr'];
    end
elseif strcmpi(ObjData.Data.Type, 'dicom_listmode') || strcmpi(ObjData.Data.Type, 'dicom_listmodelarge')
    if ObjData.Data.DCM.nListModes>1 && ObjData.Data.DCM.nListModeLarges>1
        error('Standard and Large dicom list mode files detected. Use only one of them.');
    end
    nListModeFiles = ObjData.Data.DCM.nListModes + ObjData.Data.DCM.nListModeLarges;
    if nListModeFiles>1
        fprintf('%d list-mode files were found in %s\n',nListModeFiles,FolderName)
        fprintf ('So, only static reconstructions are permitted. For dynamic reconstruction, only one list-mode file is permitted in the data folder.\n')
        
        ObjData.NumberOfFrames = 1;
        ObjData.FrameTimePoints = [];
        for i = 1: ObjData.Data.DCM.nListModes
            if strcmpi(ObjData.Data.Type, 'dicom_listmode')
                ObjData.FrameTimePoints = [ObjData.FrameTimePoints;[0, ObjData.Data.DCM.listModeHdrs(i).hdr.ImageDurationSec]];
            else
                ObjData.FrameTimePoints = [ObjData.FrameTimePoints;[0, ObjData.Data.DCM.listModeLargeHdrs(i).hdr.ImageDurationSec]];
            end
            No = num2str(i-1,'%1.2d');
            ObjData.Data.emission(i).n = [Root_path Name '-LM-' No ObjData.bar Name '-LM-' No '-sino.mhdr'];% check if you need to generate these headres for every list-mode files
            ObjData.Data.emission_listmode_hdr(i).n = [Root_path Name '-LM-' No ObjData.bar Name '-LM-' No '.hdr'];
            if strcmpi(ObjData.Data.Type, 'dicom_listmode')
                % For the standard list mode file is the .l in the
                % interfile folder.
                ObjData.Data.emission_listmode(i).n = [Root_path Name '-LM-' No ObjData.bar Name '-LM-' No '.l'];
            else
                % For the large list mode file, the original bf file in the
                % dicom folder is used:
                ObjData.Data.emission_listmode(i).n = ObjData.Data.DCM.listModeLarges(i).bf;
            end
            ObjData.Data.umap(i).n            = [Root_path Name '-LM-' No ObjData.bar Name '-LM-' No '-umap.mhdr'];
            ObjData.Data.hardware_umap(i).n   = [Root_path Name '-LM-' No ObjData.bar Name '-LM-' No '-umap-hardware.mhdr'];
            ObjData.Data.scatters(i).n        = [Root_path Name '-LM-' No ObjData.bar Name '-LM-' No '-scatter.mhdr'];
            ObjData.Data.rawdata_sino(i).n    = [Root_path Name '-LM-' No ObjData.bar 'rawdata_sino_' No ObjData.bar];
        end
        
    else
        if strcmpi(ObjData.Data.Type, 'dicom_listmode')
            ImageDurationSec = ObjData.Data.DCM.listModeHdrs(1).hdr.ImageDurationSec;
        else
            ImageDurationSec = ObjData.Data.DCM.listModeLargeHdrs(1).hdr.ImageDurationSec;
        end
        if isempty(ObjData.FrameTimePoints)
            ObjData.NumberOfFrames = 1;
            ObjData.FrameTimePoints = [0, ImageDurationSec];
        else
            ObjData.NumberOfFrames = length(ObjData.FrameTimePoints)-1;
            if sum(ObjData.FrameTimePoints) > ImageDurationSec
                error('Total frame duration > ImageDuration')
            end
        end
        
        for i = 1: ObjData.NumberOfFrames
            
            No = num2str(0,'%1.2d');
            Noi = num2str(i-1,'%1.1d');
            ObjData.Data.emission(i).n = [Root_path Name '-LM-' No ObjData.bar Name '-LM-' No '-sino-' Noi '.mhdr'];
            ObjData.Data.emission_listmode_hdr(i).n = [Root_path Name '-LM-' No ObjData.bar Name '-LM-' No '.hdr'];
            if strcmpi(ObjData.Data.Type, 'dicom_listmode')
                ObjData.Data.emission_listmode(i).n = [Root_path Name '-LM-' No ObjData.bar Name '-LM-' No '.l'];
            else
                 % For the large list mode file, the original bf file in the
                % dicom folder is used:
                ObjData.Data.emission_listmode(i).n = ObjData.Data.DCM.listModeLargeHdrs(i).bf;
            end
            ObjData.Data.umap(i).n            = [Root_path Name '-LM-' No ObjData.bar Name '-LM-' No '-umap.mhdr'];
            ObjData.Data.hardware_umap(i).n   = [Root_path Name '-LM-' No ObjData.bar Name '-LM-' No '-umap-hardware.mhdr'];
            ObjData.Data.scatters(i).n        = [Root_path Name '-LM-' No ObjData.bar Name '-LM-' No '-scatter.mhdr'];
            ObjData.Data.rawdata_sino(i).n    = [Root_path Name '-LM-' No ObjData.bar 'rawdata_sino_' Noi ObjData.bar];
            
            % make mhdr files
            ObjData.make_mhdr(ObjData.Data.emission(i).n)
        end
        
    end
    % Don' t histofram by default
%     % Check if the interfile file already exists, and if it doesn't call
%     % histogram replay:
%     if ~exist(ObjData.Data.emission(i).n, 'file')
%         fprintf('Histogram replay\n');
%         ObjData.histogram_data();
%     end
    % With the interfile list mode data ready now switch the data type to interfile list-mode:
    %ObjData.Data.Type = 'interfle_listmode';
end





