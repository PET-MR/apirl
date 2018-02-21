classdef PETDataClass < handle
    
    % This class reads the raw data files and generates the dynamic or
    % static sinograms. Data types are interfile and dicom, list-mode or
    % sinograms
    
    % % example for dicom_listmode files
    % data = PETDataClass('\\Bioeng202-pc\pet-m\FDG_Patient_02\e7\data_test') 
    % % get dynamic sinograms
    % data.frameListmodeData([0,40,100,500]);
    % prompts= data.Prompts(4);
    
    % % example for sinogram_uncompressed_interfile
    % data = PETDataClass('\\Bioeng202-pc\pet-m\Phantoms\Brain68')
    % prompts= data.Prompts;
    
    % defaults
    % 1) if the data folder contains both sinograms and list-mode files in dicom and interfile,
    % there selected data type is of the following order: 'dicom_sinogram', 'dicom_listmode',
    % 'sinogram_uncompressed_interfile', 'sinogram_interfile', 'list_mode_interfile'
    
    % 2) if there is one list-mode file in the data folder, multiple
    % dynamic sinograms is permitted
    
    % 3) if there are multiple listmodes in the data folder, only one
    % static sinogram per list-file is permitted.
    
    % if raw sinogram data already exit in some folder XXX, use the
    % following to get 3D sinograms
    % data = PETDataClass ('XXX','get_sino_rawdata'); 
    % e.g. scatters = data.Scatters; AN = data.AN;
    
    properties (SetAccess = private)
        os
        bar
        Data
        image_size
        sinogram_size
        span
        Gantry
        % A structure with fields: .span, .maxRingDiff ,.nCrystalRings
        
        MethodSinoData %'e7', 'stir'.
        % Use e7 tools to generate all sinogram data or the combination of
        % STIR ans Apirl
        
        MethodListData % 'e7', 'matlab'
        % Use e7 tools to process list mode data or a matlab version (to be
        % developed
        
        % Number of frames.
        NumberOfFrames
        
        % List with the frames time stamps (it has NumberOfFrames elements):
        DynamicFrames_sec
        
        % List with motion frames time stamps (it has NumberOfFrames elements):
        MotionFrames_sec
        
        % Frame durations (it has NumberOfFrames elements):
        FrameDuration_sec
        
        SoftwarePaths
        % A structure with fields: .e7.siemens, .e7.JSRecon12, .STIR, .Apirl
        
        
        ScanDuration_sec
        FrameTimePoints % [0,300,400,300]
        Version = '0.2.1'   % Versioning Major.Minor.Rev
    end
    
    methods
        % Constructurs
        function ObjData = PETDataClass(varargin)
            % Set default properties
            ObjData.Data.path               = ''; % Main path.
            ObjData.Data.emission           = '';
            ObjData.Data.emission_listmode  = '';
            ObjData.Data.norm               = '';
            ObjData.Data.umap               = '';
            ObjData.Data.hardware_umap      = '';
            ObjData.Data.scatters           = '';
            ObjData.Data.rawdata_sino       = '';
            
            ObjData.Data.Type                = '';
            ObjData.Data.DCM.nSinograms = 0;
            ObjData.Data.DCM.sinogramHdrs = [];
            ObjData.Data.DCM.nListModes = 0;
            ObjData.Data.DCM.listModeHdrs = [];
            ObjData.Data.DCM.nListModeLarges = 0;
            ObjData.Data.DCM.listModeLargeHdrs = [];
            ObjData.Data.DCM.nNormFiles =0;
            ObjData.Data.DCM.NormFileHdrs = [];
            ObjData.Data.IF.nCompressedSinogramFiles = 0;
            ObjData.Data.IF.CompressedSinogramHdrs = [];
            ObjData.Data.IF.nUncompressedSinogramFiles = 0;
            ObjData.Data.IF.UncompressedSinogramHdrs = [];
            ObjData.Data.IF.nListModeFiles = 0;
            ObjData.Data.IF.ListModeHdrs = [];
            ObjData.Data.IF.nNormFiles = 0;
            ObjData.Data.IF.NormFileHdrs = [];
            ObjData.Data.IF.nHumanUmaps = 0;
            ObjData.Data.IF.HumanUmapHdrs = [];
            ObjData.Data.IF.nHardwareUmaps = 0;
            ObjData.Data.IF.HardwareUmapHdrs = [];
            ObjData.Data.isListMode         =0;
            ObjData.Data.isSinogram         =0;
            
            ObjData.span                        = 11; % span by default.
            ObjData.Gantry.nCrystalRings        = 64; % number of crystal rings.
            ObjData.Gantry.nCrystalsPerRing     = 504;
            ObjData.Gantry.maxRingDiff          = 60; % maximum ring difference.
            ObjData.image_size.matrixSize       = [344, 344, 127]; % Image size by default.
            ObjData.sinogram_size.matrixSize    = [344 252,837];
            ObjData.image_size.voxelSize_mm     = [2.08626 2.08626 2.03125]; % Image size by default.
            ObjData.MethodSinoData              = 'e7';
            ObjData.MethodListData              = 'e7'; % matlab
            ObjData.DynamicFrames_sec           = []; % 0,300,200,1000,...
            ObjData.MotionFrames_sec            = [];
            ObjData.FrameDuration_sec           = 3600; %sec.
            
            ObjData.SoftwarePaths.STIR          = '';
            ObjData.SoftwarePaths.Apirl         = '';
            ObjData.SoftwarePaths.WinePath      = [getenv('HOME') '/.wine/drive_c/'];
            
            ObjData.ScanDuration_sec            =[];
            ObjData.FrameTimePoints             = [];
            ObjData.NumberOfFrames              = 1;
            
            if ObjData.span~=11
                init_sinogram_size(ObjData, ObjData.span, ObjData.Gantry.nCrystalRings, ObjData.Gantry.maxRingDiff);
            end
            if(strcmp(computer(), 'GLNXA64')) % If linux, call wine64.
                ObjData.os = 'linux';
                ObjData.bar = '/';
                ObjData.SoftwarePaths.e7.siemens   = ['wine64 ' ObjData.SoftwarePaths.WinePath '/Siemens/PET/bin.win64/e7_recon.exe'];
                ObjData.SoftwarePaths.e7.HistogramReplay   = ['wine64 ' ObjData.SoftwarePaths.WinePath '/Siemens/PET/bin.win64/HistogramReplay.exe'];
                ObjData.SoftwarePaths.e7.JSRecon12 = ['wine64 cscript ' ObjData.SoftwarePaths.WinePath '/JSRecon12/JSRecon12.js'];
            else
                ObjData.os = 'windows';
                ObjData.bar = '\';
                ObjData.SoftwarePaths.e7.siemens   = 'C:\Siemens\PET\bin.win64-VA20\e7_recon.exe';
                ObjData.SoftwarePaths.e7.HistogramReplay   = 'C:\Siemens\PET\bin.win64-VA20\HistogramReplay.exe';
                ObjData.SoftwarePaths.e7.JSRecon12 = 'cscript C:\JSRecon12\JSRecon12.js ';
            end
            
            
            % If not any path, ask for it.
            if nargin == 0
                % prompt a window to select the data path
                ObjData.Data.path  = uigetdir([],'Select the folder contaning the data');
                ObjData = ObjDataClass(ObjData.Data.path );
            elseif nargin == 2
                if strcmpi(varargin{2},'get_sino_rawdata') 
                    % only used if the sino_rawdata folder already exists somewhere
                    
                    if isdir(varargin{1})
                        ObjData.Data.path  = varargin{1};
                        ObjData.Data.rawdata_sino(1).n = ObjData.Data.path;
                        ObjData.Data.isSinogram = 1;
                    end
                else
                    error('unknown switch %s\n',varargin{2})
                end
                
            else
                % Path or struct received.
                if isstruct(varargin{1})
                    vfields = fieldnames(varargin{1});
                    prop = properties(ObjData);
                    for i = 1:length(vfields)
                        % if is an struct, update only the received fields in that
                        % structure, for example for sinogram_size:
                        if isstruct(varargin{1}.(vfields{i}))
                            field = vfields{i};
                            if sum(strcmpi(prop, field )) > 0
                                subprop = fieldnames(ObjData.(field));
                                secondary_fields = fieldnames(varargin{1}.(field));
                                for j = 1 : length(secondary_fields)
                                    subfield = secondary_fields{j};
                                    if sum(strcmpi(subprop, subfield )) > 0
                                        ObjData.(field).(secondary_fields{j}) = varargin{1}.(field).(secondary_fields{j});
                                    end
                                end
                            end
                        else
                            % It isnt
                            field = vfields{i};
                            if sum(strcmpi(prop, field )) > 0
                                ObjData.(field) = varargin{1}.(field);
                            end
                        end
                    end
 
                elseif isdir(varargin{1})
                    ObjData.Data.path  = varargin{1};
                end
                % Check for optional parameters:
                % default sinogram size:
%                 ObjData.init_sinogram_size(11, 64, 60);
                %Update ObjData.SinogramSize based on Gantry.span and Gantry.maxRingDiff
                init_sinogram_size(ObjData, ObjData.span, ObjData.Gantry.nCrystalRings, ObjData.Gantry.maxRingDiff);
                
                % main calls are here
                % if the path dosen't ends with a bar
                if ~strcmp(ObjData.Data.path(end),ObjData.bar)
                    ObjData.Data.path = [ObjData.Data.path ObjData.bar];
                end
                ParseObjDataFile(ObjData,ObjData.Data.path);
                
                if strcmpi(ObjData.Data.Type, 'sinogram_interfile')
                    uncompress_emission(ObjData);
                    ParseObjDataFile(ObjData,ObjData.Data.path);
                    ObjData.Data.Type ='sinogram_uncompressed_interfile';
                    ObjData = read_histogram_interfiles(ObjData, ObjData.Data.path);
                elseif strcmpi(ObjData.Data.Type, 'sinogram_uncompressed_interfile')
                    
                    ObjData = read_histogram_interfiles(ObjData, ObjData.Data.path);
                    
                elseif strcmpi(ObjData.Data.Type, 'dicom_sinogram') || strcmpi(ObjData.Data.Type, 'dicom_listmode') || strcmpi(ObjData.Data.Type, 'dicom_listmodelarge')
                    fprintf('Calling JSRecon12...\n');
                    ObjData = prompt_JSRecon12(ObjData, ObjData.Data.path);
                else
                    ObjData = read_histogram_interfiles(ObjData, ObjData.Data.path);
                end
            end
        end
    end
    
    
    methods (Access = private)
        
        make_mhdr(ObjData,filename);
        ObjData = prompt_JSRecon12(ObjData, FolderName);
        ObjData = read_histogram_interfiles(ObjData, FolderName,reFraming);
        % Histogram data: converts a list mode acquisition into span-1 sinograms.
        ObjData = histogram_data(ObjData);
        
        function type = ParseObjDataFile(ObjData,path)
            
            printReport = 1;
            % Check for Dicom files
            outDCM = ParseDicomDataFile(path,printReport);
            
            % Check for Interfiles
            outIF = ParseInterfileDataFile(path,printReport);
            
            
            ObjData.Data.DCM.nSinograms   = outDCM.nSinograms;
            ObjData.Data.DCM.sinogramHdrs = outDCM.SinogramHdrs;
            ObjData.Data.DCM.nListModes   = outDCM.nListModes;
            ObjData.Data.DCM.listModeHdrs = outDCM.listModeHdrs;
            ObjData.Data.DCM.nListModeLarges   = outDCM.nListModeLarges;
            ObjData.Data.DCM.listModeLargeHdrs = outDCM.listModeLargeHdrs;
            ObjData.Data.DCM.nNormFiles   = outDCM.nNormFiles;
            ObjData.Data.DCM.NormFileHdrs = outDCM.NormFileHdrs;
            
            ObjData.Data.IF.nCompressedSinogramFiles = outIF.nCompressedSinogramFiles;
            ObjData.Data.IF.CompressedSinogramHdrs = outIF.CompressedSinogramHdrs;
            ObjData.Data.IF.CompressedSinogramMhdrs = outIF.CompressedSinogramMhdrs;
            ObjData.Data.IF.nUncompressedSinogramFiles = outIF.nUncompressedSinogramFiles;
            ObjData.Data.IF.UncompressedSinogramHdrs = outIF.UncompressedSinogramHdrs;
            ObjData.Data.IF.UncompressedSinogramMhdrs = outIF.UncompressedSinogramMhdrs;
            ObjData.Data.IF.nListModeFiles = outIF.nListModeFiles;
            ObjData.Data.IF.ListModeHdrs = outIF.ListModeHdrs;
            ObjData.Data.IF.nNormFiles = outIF.nNormFiles;
            ObjData.Data.IF.NormFileHdrs = outIF.NormFileHdrs;
            ObjData.Data.IF.nHumanUmaps = outIF.nHumanUmaps;
            ObjData.Data.IF.HumanUmapHdrs = outIF.HumanUmapHdrs;
            ObjData.Data.IF.HumanUmapMhdrs = outIF.HumanUmapMhdrs;
            ObjData.Data.IF.nHardwareUmaps = outIF.nHardwareUmaps;
            ObjData.Data.IF.HardwareUmapHdrs = outIF.HardwareUmapHdrs;
            ObjData.Data.IF.HardwareUmapMhdrs = outIF.HardwareUmapMhdrs;
            % If all data types are present in the path, the prefered type
            % is as following, otherwise, the data should be seperated
            
            % Load all the filenames that were found, then select the data
            % type based on priorities:
            if outDCM.nSinograms
                ObjData.Data.Type = 'dicom_sinogram';
                ObjData.Data.isSinogram = ObjData.Data.DCM.nSinograms;
            elseif outDCM.nListModes
                ObjData.Data.isListMode = ObjData.Data.DCM.nListModes;
            elseif outDCM.nListModeLarges
                ObjData.Data.isListModeLarge = ObjData.Data.DCM.nListModeLarges;
            elseif outIF.nUncompressedSinogramFiles
                ObjData.Data.isSinogram = ObjData.Data.IF.nUncompressedSinogramFiles;
            elseif outIF.nCompressedSinogramFiles
                ObjData.Data.isSinogram = ObjData.Data.IF.nCompressedSinogramFiles;
            elseif outIF.nListModeFiles
                ObjData.Data.isListMode = ObjData.Data.IF.nListModeFiles;
            end
            if outDCM.nSinograms
                ObjData.Data.Type = 'dicom_sinogram';
                [ObjData.Data.path_raw_data, Name] = fileparts(ObjData.Data.emission(1:end-1));
                ObjData.Data.path_raw_data = [ObjData.Data.path(1:end-1) '-Converted' ObjData.bar];
            elseif outDCM.nListModes
                ObjData.Data.Type = 'dicom_listmode';
                ObjData.Data.path_raw_data = [ObjData.Data.path(1:end-1) '-Converted' ObjData.bar];
            elseif outDCM.nListModeLarges
                ObjData.Data.Type = 'dicom_listmodelarge';
                ObjData.Data.path_raw_data = [ObjData.Data.path(1:end-1) '-Converted' ObjData.bar];
            elseif outIF.nUncompressedSinogramFiles
                ObjData.Data.Type = 'sinogram_uncompressed_interfile';
                [ObjData.Data.path_raw_data, Name] = fileparts(ObjData.Data.emission(1:end-1));
                ObjData.Data.path_raw_data = ObjData.Data.path;
            elseif outIF.nCompressedSinogramFiles
                ObjData.Data.Type = 'sinogram_interfile';
                [ObjData.Data.path_raw_data, Name] = fileparts(ObjData.Data.emission(1:end-1));
                ObjData.Data.path_raw_data = ObjData.Data.path;
            elseif outIF.nListModeFiles
                ObjData.Data.Type = 'list_mode_interfile';
                [ObjData.Data.path_raw_data, Name] = fileparts(ObjData.Data.emission_listmode(1:end-1));
                ObjData.Data.path_raw_data = ObjData.Data.path;
            end
            type = ObjData.Data.Type;
            fprintf('Selected data type: %s\n',ObjData.Data.Type)
            
            
        end
        % This functions get the field of the struct used in Apirl and
        % converts to the struct sinogram_size used in this framework.
        function set_sinogram_size_from_apirl_struct(objPETRawData, structSizeSino)
            objPETRawData.sinogram_size.nRadialBins = structSizeSino.numR;
            objPETRawData.sinogram_size.nAnglesBins = structSizeSino.numTheta;
            objPETRawData.sinogram_size.nSinogramPlanes = sum(structSizeSino.sinogramsPerSegment);
            objPETRawData.sinogram_size.nPlanesPerSeg = structSizeSino.sinogramsPerSegment;
            objPETRawData.sinogram_size.matrixSize = [objPETRawData.sinogram_size.nRadialBins objPETRawData.sinogram_size.nAnglesBins objPETRawData.sinogram_size.nSinogramPlanes];
            objPETRawData.sinogram_size.span = structSizeSino.span;
            objPETRawData.sinogram_size.nRings = structSizeSino.numZ;
            objPETRawData.sinogram_size.nSeg = numel(structSizeSino.sinogramsPerSegment);
            objPETRawData.sinogram_size.minRingDiffs = structSizeSino.minRingDiffs;
            objPETRawData.sinogram_size.maxRingDiffs = structSizeSino.maxRingDiffs;
            objPETRawData.sinogram_size.numPlanesMashed = structSizeSino.numPlanesMashed;
        end
        
        function [status, message] = e7_sino_rawdata(ObjData,frame)
            % e7_sino_rawdata calls e7_recon to gereate all raw data and
            command = [ObjData.SoftwarePaths.e7.siemens ' -e "' ObjData.Data.emission(frame).n '"' ...
                ' -u "' ObjData.Data.umap(1).n '","' ObjData.Data.hardware_umap(1).n '"' ...
                ' -n "' ObjData.Data.norm '" --os "' ObjData.Data.scatters(frame).n '" --rs --force -l 73,. -d ' ObjData.Data.rawdata_sino(frame).n ];
            
            [status,message] = system(command);
            
        end
        
        function [status, message] = e7_recon(ObjData, frame, numSubsets, numIterations, enablePsf, saveIterations)
            if enablePsf
                strPsf = ' --psf ' ;
                outputTag = '-PSF';
            else
                strPsf = ' ';
                outputTag = '';
            end
            if saveIterations
                strIntermediateIters = ' --d2 ' ;
            else
                strIntermediateIters = ' ';
            end
            lastSlash = strfind(ObjData.Data.emission.n,ObjData.bar);
            filename = [ObjData.Data.emission.n(lastSlash(end)+1:end-11) outputTag];
            outputPath = [ObjData.Data.emission.n(1:lastSlash(end)) ObjData.bar 'recon' outputTag ObjData.bar];
            if ~isdir(outputPath)
                mkdir(outputPath);
            end
            fullFilename = [outputPath filename];
            command = [ObjData.SoftwarePaths.e7.siemens ' --algo op-osem --is '  num2str(numIterations) ',' num2str(numSubsets) strPsf ' -e "' ObjData.Data.emission.n '"' ...
                ' --oi "' fullFilename ' "' ' -u "' ObjData.Data.umap(1).n '","' ObjData.Data.hardware_umap(1).n '"' ...
                ' -n "' ObjData.Data.norm '"' ' --gf --quant 1 -w 344 -l 73,. --fl --ecf --izoom 1 --force --cvrg 97 --rs ' strIntermediateIters];
            [status,message] = system(command);            
        end
        
        function status = uncompress_emission(ObjData)
            % calls e7 intfcompr.exe to uncompress data
            pathstr = fileparts(ObjData.SoftwarePaths.e7.siemens);
            % update UncompressedSinogramHdrs
            % loop over the number of sinograms
            for i = 1: ObjData.Data.IF.nCompressedSinogramFiles
                compressedMhdr = [ObjData.Data.IF.CompressedSinogramHdrs(i).hdr.NameOfDataFile(1:end-2) '.mhdr'];
                UncompressedMhdr = [ObjData.Data.IF.CompressedSinogramHdrs(i).hdr.NameOfDataFile(1:end-2) '_uncomp.mhdr'];
                if ~exist(compressedMhdr,'file'), error('could not find %s\n',compressedMhdr); end
                
                command = [pathstr '\intfcompr.exe -e "' compressedMhdr '" --oe "' UncompressedMhdr '"'];
                [status,~] = system(command);
                if status
                    error('intfcompr:: uncompression was failed');
                end
            end
            out = ParseInterfileDataFile(ObjData.Data.path,0);
            ObjData.Data.IF.nUncompressedSinogramFiles = out.nUncompressedSinogramFiles;
            ObjData.Data.IF.UncompressedSinogramHdrs = out.UncompressedSinogramHdrs;
            
        end
        
        function frame = check_load_sinogram_inputs(ObjData,frame)
            if frame==0
                if ObjData.Data.isSinogram
                    if ObjData.Data.isSinogram==1
                        frame =1;
                    else
                        error('There are %d sinograms, please specify the sinogram number. \n',ObjData.Data.isSinogram)
                    end
                elseif ObjData.Data.isListMode
                    if ObjData.Data.isListMode==1
                        if  ObjData.NumberOfFrames==1
                            frame =1;
                        else
                            error('There are %d dynamic frames, please specify the frame number.\n',ObjData.NumberOfFrames)
                        end
                    elseif ObjData.Data.isListMode>1
                        error('There are %d list-mode files, please specify the static sinogram number.\n',ObjData.Data.isListMode)
                    end
                elseif ObjData.Data.isListModeLarge
                    if ObjData.Data.isListModeLarge==1
                        if  ObjData.NumberOfFrames==1
                            frame =1;
                        else
                            error('There are %d dynamic frames, please specify the frame number.\n',ObjData.NumberOfFrames)
                        end
                    elseif ObjData.Data.isListModeLarge>1
                        error('There are %d list-mode large format files, please specify the static sinogram number.\n',ObjData.Data.isListModeLarge)
                    end
                end
            else
                if ObjData.Data.isSinogram==1 && frame>1
                    fprintf('The requested sinogram number exceeds the number of available sinograms, loading the last one (%d)\n', ObjData.Data.isSinogram);
                    frame = 1;
                elseif ObjData.Data.isSinogram >1 && frame>ObjData.Data.isSinogram % multiple sinograms in a folder
                    fprintf('The requested sinogram number exceeds the number of available sinograms, loading the last one (%d)\n', ObjData.Data.isSinogram);
                    frame = ObjData.Data.isSinogram;
                elseif ObjData.Data.isListMode==1 && frame>ObjData.NumberOfFrames
                    fprintf('The requested dynamic frame number exceeds the number of frames, loading the last one (%d)\n', ObjData.NumberOfFrames);
                    frame = ObjData.NumberOfFrames;
                elseif ObjData.Data.isListMode>1 && frame>ObjData.Data.isListMode
                    fprintf('The requested static sinogram number exceeds the number of list-mode files, loading the last one (%d)\n', ObjData.Data.isListMode);
                    frame = ObjData.Data.isListMode;
                end
            end
        end
        
        function data = Load_sinogram(ObjData,sType,frame)
            % if the requested sinogram exists in ObjData.Data.rawdata_sino,
            % Load_sinogram() loads it, otherwise it calls the relevent functions (based on MethodSinoData)
            % to generate the sinogram, waits until the sinogram is
            % generated and then loads it.
            
            % todo: Axial compression, if requested
            if ~exist(ObjData.Data.rawdata_sino(frame).n,'dir') || length(dir(ObjData.Data.rawdata_sino(frame).n))==2
                mkdir(ObjData.Data.rawdata_sino(frame).n);
                if strcmpi(ObjData.MethodSinoData,'e7') && (ObjData.span ==11)
                    fprintf('Calling e7_recon...');
                    [status, message] = e7_sino_rawdata(ObjData,frame);
                    if status
                        error(['e7_recon failed to generate sinograms: ' message]);
                    end
                    fprintf('Done.\n');
                else % STIR version
                    
                end
                
            end
            
            switch sType
                % The nomenclature and extension of sinograms follows
                % the e7 tools, the output of STIR and Apirl should follow
                % the same or another if statement should be used with a
                % similar switch-case statement.
                case 'prompts'
                    filename  = [ObjData.Data.rawdata_sino(frame).n ObjData.bar 'emis_00.s'];
                    data = read_sinograms(ObjData,filename, ObjData.sinogram_size.matrixSize);
                case 'randoms'
                    filename  = [ObjData.Data.rawdata_sino(frame).n ObjData.bar 'smoothed_rand_00.s'];
                    data = read_sinograms(ObjData,filename, ObjData.sinogram_size.matrixSize);
                case 'NCF'
                    filename  = [ObjData.Data.rawdata_sino(frame).n ObjData.bar 'norm3d_00.a'];
                    data = read_sinograms(ObjData,filename, ObjData.sinogram_size.matrixSize);
                case 'ACF'
                    filename  = [ObjData.Data.rawdata_sino(frame).n ObjData.bar 'acf_00.a'];
                    data = read_sinograms(ObjData,filename, ObjData.sinogram_size.matrixSize);
                case 'ACF2'
                    filename  = [ObjData.Data.rawdata_sino(frame).n ObjData.bar 'acf_second_00.a'];
                    data = read_sinograms(ObjData,filename, ObjData.sinogram_size.matrixSize);
                case 'scatters'
                    filename  = [ObjData.Data.rawdata_sino(frame).n ObjData.bar 'scatter_estim2d_000000.s'];
                    scatter_2D = read_sinograms(ObjData,filename, [ObjData.sinogram_size.matrixSize(1:2) 127]);
                    scatter_3D = iSSRB(ObjData,scatter_2D);
                    % 2D scatters generated by e7 tools are already scaled
                    %  data = scatter_scaling(ObjData,scatter_3D,frame);
                    data = scatter_3D;
                    clear scatter_3D scatter_2D
            end
        end
        
        function data = read_sinograms(~,filename, SinoSize)
            fid=fopen(filename,'r');
            d = fread(fid,prod(SinoSize),'float');
            data = single(reshape(d,SinoSize));
            fclose(fid);
        end
        
        function Scatter3D = iSSRB(ObjData,Scatter2D)
            
            nPlanePerSeg = Planes_Seg(ObjData);
            
            mo = cumsum(nPlanePerSeg)';
            no =[[1;mo(1:end-1)+1],mo];
            
            Scatter3D = zeros(ObjData.sinogram_size.matrixSize,'single');
            Scatter3D(:,:,no(1,1):no(1,2),:) = Scatter2D;
            
            for i = 2:2:length(nPlanePerSeg)
                
                delta = (nPlanePerSeg(1)- nPlanePerSeg(i))/2;
                indx = nPlanePerSeg(1) - delta;
                
                Scatter3D (:,:,no(i,1):no(i,2),:) = Scatter2D(:,:,delta+1:indx,:);
                Scatter3D (:,:,no(i+1,1):no(i+1,2),:) = Scatter2D(:,:,delta+1:indx,:);
            end
        end
        
        function scatter_3D = scatter_scaling(ObjData,scatter_3D,frame)
            P = ObjData.Prompts(frame);
            R = ObjData.Randoms(frame);
            NCF = ObjData.NCF(frame);
            ACF2 = ObjData.ACF2(frame);
            
            gaps = NCF==0;
            scatter_3D =  scatter_3D./NCF;
            scatter_3D(gaps)=0;
            
            Trues = P - R;
            
            for i = 1: size(ACF2,3)
                acf_i = ACF2(:,:,i);
                mask = acf_i <= min(acf_i(:));
                
                scatter_3D_tail = scatter_3D(:,:,i).*mask;
                Trues_tail = Trues(:,:,i).*mask;
                
                scale_factor = sum(Trues_tail(:))./sum(scatter_3D_tail(:));
                scatter_3D(:,:,i) = scatter_3D(:,:,i)*scale_factor;
            end
            % normalizations are now taken into account in ObjData.Scatters, 
            % as scatter_scaling might be skipped as with e7 data
            scatter_3D =  scatter_3D.*NCF; 
            clear NCF ACF2 gaps R P Trues
        end
        
    end
    
    
    
    methods (Access = public)
        
        [totalPrompts,totalRandoms, totalWords, outFileHdr, output_listmode_file] = undersample_mMR_listmode_data(ObjData,input_listmode_file,countReductionFractor,chunk_size_events, numRealizations)
        % Initialize sinogram size struct.
        sino_size_out = init_sinogram_size(ObjData, inspan, numRings, maxRingDifference);
        % Change span sinogram.
        [sinogram_out, sinogram_size_out] = change_sinogram_span(ObjData, sinogram_in, sinogram_size_in, span);
        
        function  P = Prompts(ObjData,frame)
            if nargin==1, frame = 0; end
            frame = check_load_sinogram_inputs(ObjData,frame);
            P = Load_sinogram(ObjData,'prompts',frame);
        end
        
        function  R = Randoms(ObjData,frame)
            if nargin==1, frame = 0; end
            frame = check_load_sinogram_inputs(ObjData,frame);
            R = Load_sinogram(ObjData,'randoms',frame);
        end
        
        function  S = Scatters(ObjData,frame)
            if nargin==1, frame = 0; end
            frame = check_load_sinogram_inputs(ObjData,frame);
            S = Load_sinogram(ObjData,'scatters',frame);
            nf = 1./ObjData.NCF(frame);
            nf(isinf(nf))=0;
            S = nf.*S;
        end
        
        function  ncf = NCF(ObjData,frame)
            if nargin==1, frame = 0; end
            frame = check_load_sinogram_inputs(ObjData,frame);
            ncf = Load_sinogram(ObjData,'NCF',frame);
            ncf = ncf .*ObjData.get_gaps();
        end
        
        function  acf = ACF(ObjData,frame)
            if nargin==1, frame = 0; end
            frame = check_load_sinogram_inputs(ObjData,frame);
            acf = Load_sinogram(ObjData,'ACF',frame);
        end
        
        function  acf = ACF2(ObjData,frame)
            if nargin==1, frame = 0; end
            frame = check_load_sinogram_inputs(ObjData,frame);
            acf = Load_sinogram(ObjData,'ACF2',frame);
        end
        
        function mul = AN(ObjData,frame)
            if nargin==1, frame = 0; end
            frame = check_load_sinogram_inputs(ObjData,frame);
            
            mul = ObjData.ACF(frame) .* ObjData.NCF(frame);
            mul = 1./mul;
            mul(isinf(mul)) = 0;

        end
        
        function Reconstruct(ObjData, numSubsets, numIterations, enablePsf, saveIterations)
            for frame = 1 : ObjData.NumberOfFrames
                [status, message] = e7_recon(ObjData, frame, numSubsets, numIterations, enablePsf, saveIterations);
            end
        
        end
        
        function add = RS(ObjData,frame)
            if nargin==1, frame = 0; end
            frame = check_load_sinogram_inputs(ObjData,frame);
            add = ObjData.ACF(frame) .* ObjData.NCF(frame).*(ObjData.Scatters(frame) + ObjData.Randoms(frame));
        end
        
        %         function mu = mumap(ObjData)
        %             mu = Load_image();
        %         end
        
       
        function frameListmodeData(ObjData,newFrame)

            ObjData.FrameTimePoints             = newFrame;
            ObjData.NumberOfFrames              = length(ObjData.FrameTimePoints)-1;
            
            if strcmpi(ObjData.Data.Type,'dicom_listmode') && ObjData.Data.isListMode==1
                ObjData.prompt_JSRecon12(ObjData.Data.path);
            else
                ObjData.read_histogram_interfiles(ObjData.Data.path);
            end
        end
        
        % Initalizes tima frames
        function InitFramesConfig(ObjData, timeFrame_sec)
            % Get info from the header:
            info = getInfoFromInterfile(ObjData.Data.emission_listmode(1).n);
            % Scan time:
            scanTime_sec = info.ImageDurationSec;
            % Frame durations (it has NumberOfFrames elements):
            ObjData.FrameDuration_sec = timeFrame_sec;
            % Read from the header of the list file, the total time:
            % Number of frames.
            ObjData.NumberOfFrames = ceil(scanTime_sec./timeFrame_sec);
            
            % List with the frames time stamps (it has NumberOfFrames elements):
            ObjData.DynamicFrames_sec = zeros(ObjData.NumberOfFrames,1);
            
            % Generate the time stamps for the frames:
            ObjData.DynamicFrames_sec = 0 : timeFrame_sec : timeFrame_sec*ObjData.NumberOfFrames;
        end
        
        
        function SinogramDisplay(ObjData,sType)
            if strcmpi(sType,'ACF')
                temp = ACF(ObjData,1);
            else
                temp = Prompts(ObjData,1);
            end
            figure
            for i = 1:5:ObjData.sinogram_size.matrixSize(2)
                drawnow,imshow(squeeze(temp(:,i,:)),[])
            end
            clear temp
        end
        
        function data = AxialCompress(ObjData,span)
            sinogram_in = ObjData.sinogram_size;
            ObjData.span = span;
            % sinogram_out = init_sinogram_size(ObjData, ObjData.span, ObjData.Gantry.nCrystalRings, ObjData.Gantry.maxRingDiff);
            % Compress the emission sinogram:
            [sinogram_out, sinogram_size_out] = change_sinogram_span(ObjData, sinogram, sinogram_size_in);
        end
        
        function PlotMichelogram(ObjData)
            
            Span = ObjData.span;
            MRD = ObjData.Gantry.maxRingDiff;
            nCrystalRings  = ObjData.Gantry.nCrystalRings;
            
            a = (Span+1)/2;
            b = floor((MRD +1 - a)/Span)*Span ;
            c = MRD +1 - (a+b);
            
            nSeg = 2*floor((MRD - a)/Span)+ 3;
            
            maxRingDiff = (MRD -c):-Span:-(MRD -c);
            minRingDiff = (MRD -c -Span+1):-Span:-(MRD -c);
            if c~=0
                maxRingDiff = [ MRD, maxRingDiff,-(MRD-(c-1))];
                minRingDiff = [MRD-(c-1), minRingDiff, -MRD ];
            end
            
            colo = [1:(nSeg-1)/2,(nSeg-1)/2+1,(nSeg-1)/2:-1:1];
            M = reshape(1:nCrystalRings^2,nCrystalRings,nCrystalRings);
            Michelogram = zeros(nCrystalRings);
            for j = 1:nSeg
                diagOffsetsPerSeg = minRingDiff(j):maxRingDiff(j);
                for i = 1:length(diagOffsetsPerSeg)
                    idx = diag(M,diagOffsetsPerSeg(i));
                    Michelogram(idx) = colo(j);
                end
            end
            figure,imagesc(Michelogram),colormap gray, axis xy
            title(sprintf('No. Segments:%d, Span: %d, MRD: %d', nSeg, Span, MRD))
        end
        
        
        function uncompress(ObjData,Dir)
            if nargin==1
                [name,pathstr]  = uigetfile('*.hdr');
                Dir = [pathstr name];
            end
            [pathstr,name] = fileparts(Dir);
            ObjData.Data.emission = Dir;
            ObjData.Data.emission_uncomp = [pathstr '\' name(1:end-2) '_uncomp.s.hdr'];
            uncompress_emission(ObjData);
        end
        function display(ObjData) %#ok<DISPLAY>
            disp(ObjData)
            methods(ObjData)
        end
        
        function gaps = get_gaps(ObjData)
            
            rFov_mm = 594/2; zFov_mm = 258;
            structSizeSino3d = getSizeSino3dFromSpan(ObjData.sinogram_size.matrixSize(1), ObjData.sinogram_size.matrixSize(2), ObjData.Gantry.nCrystalRings, rFov_mm, zFov_mm, ObjData.span, ObjData.Gantry.maxRingDiff);
            
            eff = ones(ObjData.Gantry.nCrystalsPerRing,ObjData.Gantry.nCrystalRings);
            eff(9:9:end,:) = 0;
            
            gaps = createSinogram3dFromDetectorsEfficency(eff, structSizeSino3d, 0);
        end
        
        function status = correct_mumap_positioning(ObjData)
            %% Read Mumap info to correct mispostioning:
            uMapInfoFilename = [ObjData.Data.path_raw_data 'UMapSeries' ObjData.bar 'UMapInfo.txt']; 
            fid = fopen(uMapInfoFilename, 'r');
            file_content_by_line = textscan(fid, '%s', inf,'Delimiter','\n');
            fclose(fid);
            % Replace the 3rd line with the content of the dicom mumap:
            list_filed = dir(ObjData.Data.path);
            for i = 3 : numel(list_filed) % first two are ., ..
               if list_filed(i).isdir
                   mumap_path = [ObjData.Data.path list_filed(i).name ObjData.bar];
               end
            end
            slices_dicom_mumap = dir(mumap_path);
            % we consider that the mumap is the first file (needs to be the first slice):
            slices_dicom_mumap(3).name;
            info_umap = dicominfo([mumap_path slices_dicom_mumap(3).name]);
            % Read dicom header:
            file_content_by_line{1}{3} = [num2str(info_umap.ImagePositionPatient(1)) ' ' num2str(info_umap.ImagePositionPatient(2)) ' ' num2str(info_umap.ImagePositionPatient(3))];
            % Write the new line, before create a back up:
            copyfile(uMapInfoFilename, [uMapInfoFilename '.bak']);
            % Now rewrite it:
            fid = fopen(uMapInfoFilename, 'w');
            for i = 1 : numel(file_content_by_line{1})
                fprintf(fid, '%s\r\n', file_content_by_line{1}{i});
            end
            fclose(fid);

            % study info:
            jsreconInfoFilename = [ObjData.Data.path_raw_data 'JSRecon12Info.txt']; 
            fid = fopen(jsreconInfoFilename, 'r');
            file_content_by_line = textscan(fid, '%s', inf,'Delimiter','\n');
            lines_resample = strfind(file_content_by_line{1}, 'volume_resample');
            for i = 1 : numel(lines_resample)
                if ~isempty(lines_resample{i})
                    spaces = strfind(file_content_by_line{1}{i}, ' ');
                    resample_command = file_content_by_line{1}{i}(spaces+1:end);
                    break;
                end
            end
            fclose(fid);
            % call the command:
            [status, message] = system(resample_command);
        end
    end
    
end


%         function path = STIR_scatter_sino()
%
%         end
%
%         function path = Apirl_norm_factors()
%
%         end
%
%         function path = e7_norm_factors()
%             % e7_norm_factors calls e7_norm to generate frame dependent norm factors in
%             % the case of dynamic or framed data
%         end


%         function Dicom2IF()
%             % converts Dicom to interfile, uses the interfile header
%             % availabe in Dicom Private_0029_1010 field
%         end
%

%
%         function Members(ObjData)
%             %displays members and defaults values
%
%         end
