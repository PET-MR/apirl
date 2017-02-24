% *********************************************************************
% Reconstruction Framework for Siemens Biograph mMR.
% class: PETRawData
% Authors: Martin Belzunce, Abolfazl Mehranian. Kings College London.
% Date: 22/02/2016
% *********************************************************************

function ObjData = read_histogram_interfiles(ObjData, FolderName)


if ObjData.Data.isSinogram
    
    for i = 1: ObjData.Data.IF.nUncompressedSinogramFiles
        UncompMhdr = ObjData.Data.IF.UncompressedSinogramMhdrs(i).hdr.NameOfMhdrFile;
        ObjData.Data.emission(i).n = UncompMhdr;
        if ~exist(UncompMhdr,'file'), error('could not find %s\n',UncompMhdr); end
        
        No = num2str(i-1,'%1.2d');
        ObjData.Data.rawdata_sino(i).n = [FolderName ObjData.bar 'rawdata_sino_' No ];
        ObjData.Data.scatters(i).n = [FolderName '\' No '-scatter.mhdr'];
    end
    for i = 1: ObjData.Data.IF.nHumanUmaps
        HuMapMhdr = ObjData.Data.IF.HumanUmapMhdrs(i).hdr.NameOfMhdrFile;
        ObjData.Data.umap(i).n = HuMapMhdr;
        if ~exist(HuMapMhdr,'file'), error('could not find %s\n',HuMapMhdr); end
    end
    for i = 1: ObjData.Data.IF.nHardwareUmaps
        HardMapMhdr = ObjData.Data.IF.HardwareUmapMhdrs(i).hdr.NameOfMhdrFile;
        ObjData.Data.hardware_umap(i).n = HardMapMhdr;
        if ~exist(HardMapMhdr,'file'), error('could not find %s\n',HardMapMhdr); end
    end
    
    if ObjData.Data.IF.nNormFiles> 1, error('more than one norm file were found\n'); end
    NormHdr = [ObjData.Data.IF.NormFileHdrs.hdr.NameOfDataFile '.hdr'];
    ObjData.Data.norm = NormHdr;
    if ~exist(NormHdr,'file'), error('could not find %s\n',NormHdr); end
     
end




if ObjData.Data.isListMode
    % The NameOfDataFile in list-mode interfile is often not correct, so
    % need to find the correct filename
    Dir = dir(ObjData.Data.path);
    nListFiles = 0;
    for i=2:length(Dir)
        fileName = [ObjData.Data.path ObjData.bar Dir(i).name];
        [~,~,ext] = fileparts(fileName);
        if strcmpi(ext,'.l') && ~isempty(ext)
            nListFiles = nListFiles + 1;
            Hdr = ObjData.Data.IF.ListModeHdrs(nListFiles).hdr;
            Hdr.NameOfDataFile = fileName;
            ObjData.Data.IF.ListModeHdrs(nListFiles).hdr = Hdr;
        end
    end


    if ObjData.Data.isListMode>1
        fprintf('%d list-mode files were found in %s\n',ObjData.Data.IF.nListModeFiles,FolderName)
        fprintf ('Hence, only full frame histogramming are permitted. For dynamic frames, only one list-mode file is permitted in the data folder.\n')
        
        ObjData.NumberOfFrames = 1;
        ObjData.FrameTimePoints = [];
        for i = 1: ObjData.Data.isListMode
            
            ObjData.FrameTimePoints = [ObjData.FrameTimePoints;[0, ObjData.Data.IF.ListModeHdrs(i).hdr.ImageDurationSec]];
            No = num2str(i-1,'%1.1d');
            
            ObjData.Data.emission(i).n = [ObjData.Data.IF.ListModeHdrs(i).hdr.NameOfDataFile(1:end-2) '-sino-0.mhdr'];
            ObjData.Data.emission_listmode_hdr(i).n = [ObjData.Data.IF.ListModeHdrs(i).hdr.NameOfDataFile(1:end-2) '.hdr']; % remove the .l
            ObjData.Data.emission_listmode(i).n = ObjData.Data.IF.ListModeHdrs(i).hdr.NameOfDataFile;
            % umap
            HuMapMhdr = ObjData.Data.IF.HumanUmapMhdrs(1).hdr.NameOfMhdrFile;
            ObjData.Data.umap(i).n = HuMapMhdr;
            if ~exist(HuMapMhdr,'file'), error('could not find %s\n',HuMapMhdr); end
            % hardware mumap
            HardMapMhdr = ObjData.Data.IF.HardwareUmapMhdrs(1).hdr.NameOfMhdrFile;
            ObjData.Data.hardware_umap(i).n = HardMapMhdr;
            if ~exist(HardMapMhdr,'file'), error('could not find %s\n',HardMapMhdr); end
            % scatters
            ObjData.Data.scatters(i).n   = [ObjData.Data.path ObjData.bar No '-scatter.mhdr'];
            ObjData.Data.rawdata_sino(i).n    = [ObjData.Data.path ObjData.bar  'rawdata_sino_' No ObjData.bar];
            % make mhdr files
            ObjData.make_mhdr(ObjData.Data.emission(i).n)
        end
        
    else
        if isempty(ObjData.FrameTimePoints)
            ObjData.NumberOfFrames = 1;
            ObjData.FrameTimePoints = [0,ObjData.Data.IF.ListModeHdrs(1).hdr.ImageDurationSec];
        else
            ObjData.NumberOfFrames = length(ObjData.FrameTimePoints)-1;
            if sum(ObjData.FrameTimePoints) > ObjData.Data.IF.ListModeHdrs(1).hdr.ImageDurationSec
                error('Total frame duration > ImageDuration')
            end
        end
        
        for i = 1: ObjData.NumberOfFrames
            No = num2str(i-1,'%1.1d');
            ObjData.Data.emission(i).n = [ObjData.Data.IF.ListModeHdrs(1).hdr.NameOfDataFile(1:end-2) '-sino-' No '.mhdr'];
            ObjData.Data.emission_listmode_hdr(i).n = [ObjData.Data.IF.ListModeHdrs(1).hdr.NameOfDataFile '.hdr'];
            ObjData.Data.emission_listmode(i).n = ObjData.Data.IF.ListModeHdrs(1).hdr.NameOfDataFile;
           
            % umap
            HuMapMhdr = ObjData.Data.IF.HumanUmapMhdrs(1).hdr.NameOfMhdrFile;
            ObjData.Data.umap(i).n = HuMapMhdr;
            if ~exist(HuMapMhdr,'file'), error('could not find %s\n',HuMapMhdr); end
            
            % hardware mumap
            HardMapMhdr = ObjData.Data.IF.HardwareUmapMhdrs(1).hdr.NameOfMhdrFile;

            ObjData.Data.hardware_umap(i).n = HardMapMhdr;
            if ~exist(HardMapMhdr,'file'), error('could not find %s\n',HardMapMhdr); end
            % scatters
            ObjData.Data.scatters(i).n   = [ObjData.Data.path ObjData.bar No '-scatter.mhdr'];
            ObjData.Data.rawdata_sino(i).n    = [ObjData.Data.path ObjData.bar  'rawdata_sino_' No ObjData.bar];
            % make mhdr files
            ObjData.make_mhdr(ObjData.Data.emission(i).n)
        end
    end
    
    if ObjData.Data.IF.nNormFiles> 0
        if ObjData.Data.IF.nNormFiles> 1
            warning('more than one norm file were found\n');
        end
        NormHdr = [ObjData.Data.IF.NormFileHdrs.hdr.NameOfDataFile '.hdr'];
        ObjData.Data.norm = NormHdr;
        if ~exist(NormHdr,'file'), warning('could not find %s\n',NormHdr); end
    end
    
    %call Histogram_replay to generate the sinograms per frame
    fprintf('Histogram replay\n');
    ObjData.histogram_data();
    
    
end