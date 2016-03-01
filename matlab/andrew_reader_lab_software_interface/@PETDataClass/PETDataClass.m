classdef PETDataClass < handle
    properties (SetAccess = private)
        % Operative system.
        os
        % bar for the paths.
        bar
        
        DataPath
        % The folder path containing raw data in Dicom IMA and PDT or Interfile
        % format. If Dicom, the folder should contain 1) emission data,
        % 2) normalization file and 3) human u-map. If Interfile, it should
        % contain 1) emission.s.hdr 2) norm.n.hdr, 3) umap_hardware.mhdr
        % and 4) umap_human.mhdr
        
        % Struct with the image size.
        image_size
        
        % Struct with sinogram size (including matrix size, number of
        % segments, etc..)
        sinogram_size
        
        % Span of the reconstructions:
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
        
        DataType
        
        Version = '0.2.1'   % Versioning Major.Minor.Rev   
    end
    
    methods
        % Constructurs
        function PETData = PETDataClass(varargin)
            % Set default properties
            PETData.DataPath.path               = ''; % Main path.
            PETData.DataPath.emission           = '';
            PETData.DataPath.emission_listmode  = '';
            PETData.DataPath.norm               = '';
            PETData.DataPath.umap               = '';
            PETData.DataPath.hardware_umap      = '';
            PETData.DataPath.rawdata_sino       = '';
            PETData.DataPath.scatters           = '';
            PETData.span                        = 11; % span by default.
            PETData.Gantry.nCrystalRings        = 64; % number of crystal rings.
            PETData.Gantry.nCrystalsPerRing     = 504;  
            PETData.Gantry.maxRingDiff          = 60; % maximum ring difference.
            PETData.image_size.matrixSize =[344, 344, 127]; % Image size by default.
            PETData.image_size.voxelSize_mm = [2.08626 2.08626 2.03125]; % Image size by default.
            % Init PETData.sinogram_size:
            init_sinogram_size(PETData, PETData.span, PETData.Gantry.nCrystalRings, PETData.Gantry.maxRingDiff); % span 11 by default.
            PETData.MethodSinoData              = 'e7';
            PETData.MethodListData              = 'e7';
            PETData.DynamicFrames_sec           = [];
            PETData.MotionFrames_sec            = [];
            PETData.FrameDuration_sec          = 3600; %sec.
            PETData.SoftwarePaths.STIR         = '';
            PETData.SoftwarePaths.Apirl        = '';        
            if(strcmp(computer(), 'GLNXA64')) % If linux, call wine.
                PETData.os = 'linux';
                PETData.bar = '/';
                PETData.SoftwarePaths.e7.siemens   = 'wine C:\Siemens\PET\bin.win64-VA20\e7_recon.exe';
                PETData.SoftwarePaths.e7.JSRecon12 = 'wine cscript C:\JSRecon12\JSRecon12.js ';
            else
                PETData.os = 'windows';
                PETData.bar = '\';
                PETData.SoftwarePaths.e7.siemens   = 'C:\Siemens\PET\bin.win64-VA20\e7_recon.exe';
                PETData.SoftwarePaths.e7.JSRecon12 = 'cscript C:\JSRecon12\JSRecon12.js ';
            end
            
            % If not any path, ask for it.
            if nargin == 0
            % prompt a window to select the data path
                PETData.DataPath.path  = uigetdir();
                PETData = PETDataClass(PETData.DataPath.path );
            else
                % Path or struct received.
            
                if isstruct(varargin{1})
                    % get fields from user's input
                    vfields = fieldnames(varargin{1});
                    prop = properties(PETData);
                    for i = 1:length(vfields)
                        field = vfields{i};
                        if sum(strcmpi(prop, field )) > 0
                            PETData.(field) = varargin{1}.(field);
                        end
                    end
                    %check the consistancy of the PETData.SinogramSize and the
                    %Gantry.span and Gantry.maxRingDiff

                elseif isdir(varargin{1})
                    PETData.DataPath.path  = varargin{1};
                    [PETData.DataType] = Which(PETData,PETData.DataPath.path );
                    switch(PETData.DataType)
                        case DataTypeEnum.sinogram_interfile
                            % Read sinogram or listmode file in interfile format and 
                            % return PETData.DataPath structure
                            PETData = read_check_interfiles(PETData, FolderName);
                        case DataTypeEnum.compressed_interfile
                            % Uncompress and update data type:
                            uncompress_emission(PETData);
                            PETData.DataType = DataTypeEnum.sinogram_interfile;
                        case DataTypeEnum.dicom
                            %  Call JSRecon12 to generate Siemens interfiles and
                            %  return PETData.DataPath structure
                            fprintf('Calling JSRecon12...\n');
                            PETData = prompt_JSRecon12(PETData, FolderName);
                        case DataTypeEnum.list_mode_interfile
                            % In PETData.DataPath.emission we have the
                            % filename. Before histograming we need the
                            % time frames.
%                             list_files = dir([PETData.DataPath.path PETData.bar '*.l.hdr']);
%                             if numel(list_files) == 0
%                                 error('Error: list mode file not found.');
%                             elseif numel(list_files) > 1
%                                 error('Error: more thean one list-mode file in the path.');
%                             else
%                                 % Set the emission filename:
%                                 PETData.DataPath.emission_listmode =[PETData.DataPath.path PETData.bar list_files(1).name];
%                             end
                            % Check for the other files:
                            PETData = read_check_interfiles(PETData, PETData.DataPath.path);
                           %PETData = histogram_data(PETData, lists, SinogramSize);
                    end
                else

                end
                % Check for optional parameters:
                if nargin == 2
                    if isstruct(varargin{2})
                        % get fields from user's input
                        vfields = fieldnames(varargin{2});
                        prop = properties(PETData);
                        for i = 1:length(vfields)
                            field = vfields{i};
                            field_value = varargin{2}.(field);
                            if strcmpi(field,'span')
                                PETData.span = field_value;
                            elseif strcmpi(field,'maxRingDiff')
                                PETData.Gantry.maxRingDiff = field_value;
                            else
                                if sum(strcmpi(prop, field )) > 0
                                    PETData.(field) = field_value;
                                end
                            end
                        end

                        %Update PETData.SinogramSize based on Gantry.span and Gantry.maxRingDiff
                        init_sinogram_size(PETData, PETData.span, PETData.Gantry.nCrystalRings, PETData.Gantry.maxRingDiff); 
                    end
                end
            end
        end
    end
    
    
    methods (Access = private)
        % Initialize sinogram size struct.
        sino_size_out = init_sinogram_size(PETData, inspan, numRings, maxRingDifference);
        % Change span sinogram.
        [sinogram_out, sinogram_size_out] = change_sinogram_span(PETData, sinogram_in, sinogram_size_in);
        
        PETData = prompt_JSRecon12(PETData, FolderName);
        PETData = read_check_interfiles(PETData, FolderName);
        % Histogram data: converts a list mode acquisition into span-1 sinograms.
        PETData = histogram_data(PETData);
        
        function [type] = Which(PETData,path)
            % In the folder only one type of data files can be available.
            PETData.DataType = DataTypeEnum.none;
            listing = dir(path);
            for i = 3:length(listing)
                [path, name, ext] = fileparts(listing(i).name);
                if strcmpi(ext,'.IMA') || strcmpi(ext,'.PDT')
                    PETData.DataType = DataTypeEnum.dicom;
                    if (PETData.DataType ~=  DataTypeEnum.none) && (PETData.DataType ~=  DataTypeEnum.dicom)
                        error('Please seperate the dicom files from the interfile files.')
                    end
                else
                    if strcmpi(ext,'.mhdr')
                        PETData.DataType = DataTypeEnum.sinogram_interfile;
                        if (PETData.DataType ~=  DataTypeEnum.none) && (PETData.DataType ~=  DataTypeEnum.sinogram_interfile)
                            error('Please seperate the sinograms in interfile from list mode and dicom files.')
                        end
                    elseif strcmpi(ext,'.hdr')
                        [dummy,name,ext] = fileparts(name);
                        if strcmpi(ext,'.s') 
                            PETData.DataType = DataTypeEnum.sinogram_interfile;
                            if (PETData.DataType ~=  DataTypeEnum.none) && (PETData.DataType ~=  DataTypeEnum.sinogram_interfile)
                                error('Please seperate the sinograms in interfile from list mode and dicom files.')
                            end
                        elseif strcmpi(ext,'.l') 
                            PETData.DataType = DataTypeEnum.list_mode_interfile;
                            if (PETData.DataType ~=  DataTypeEnum.none) && (PETData.DataType ~=  DataTypeEnum.list_mode_interfile)
                                error('Please seperate the list_mode files from sinograms and dicom files.')
                            end
                        end
                    end
                end
            end
            if PETData.DataType == DataTypeEnum.none
                error('No Dicom/Interfile was found.');
            end       
            type = PETData.DataType;
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
        
        function status = e7_sino_rawdata(PETData)
            % e7_sino_rawdata calls e7_recon to gereate all raw data and
            command = [PETData.SoftwarePathes.e7.siemens ' -e ' PETData.DataPath.emission ...
                ' -u ' PETData.DataPath.umap ',' PETData.DataPath.hardware_umap...
                ' -n ' PETData.DataPath.norm ' --os ' PETData.DataPath.scatters ' --rs --force -l 73,. -d ' PETData.DataPath.rawdata_sino];
            [status,~] = system(command);
        end
        
        function status = uncompress_emission(PETData)
            % calls e7 intfcompr.exe to uncompress data
            command = [PETData.SoftwarePathes.e7.siemens 'intfcompr.exe -e ' PETData.DataPath.emission ' --oe ' PETData.DataPath.emission_uncomp];
            [status,~] = system(command);
            if status
                error('intfcompr:: uncompression was failed');
            end
        end
        
        function path = STIR_scatter_sino()
            
        end
        
        function path = Apirl_norm_factors()
            
        end
        
        function path = e7_norm_factors()
            % e7_norm_factors calls e7_norm to generate frame dependent norm factors in
            % the case of dynamic or framed data
        end
        
        function data = Load_sinogram(PETData,sType)
            % if the requested sinogram exists in PETData.DataPath.rawdata_sino,
            % Load_sinogram() loads it, otherwise it calls the relevent functions (based on MethodSinoData)
            % to generate the sinogram, waits until the sinogram is
            % generated and then loads it.
            
            % todo: Axial compression, if requested
            if ~exist(PETData.DataPath.rawdata_sino,'dir')
                mkdir(PETData.DataPath.rawdata_sino);
                if strcmpi(PETData.MethodSinoData,'e7') && (PETData.Gantry.span ==11)
                    status = e7_sino_rawdata(PETData);
                    if status
                        error('e7_recon failed to generate sinograms');
                    end
                else % STIR version
                    
                end
            end
            
            switch sType
                % The nomenclature and extension of sinograms follows
                % the e7 tools, the output of STIR and Apirl should follow
                % the same or another if statement should be used with a
                % similar switch-case statement.
                case 'prompts'
                    filename  = [PETData.DataPath.rawdata_sino '\emis_00.s'];
                    data = read_sinograms(PETData,filename, PETData.SinogramSize);
                case 'randoms'
                    filename  = [PETData.DataPath.rawdata_sino '\smoothed_rand_00.s'];
                    data = read_sinograms(PETData,filename, PETData.SinogramSize);
                case 'NCF'
                    filename  = [PETData.DataPath.rawdata_sino '\norm3d_00.a'];
                    data = read_sinograms(PETData,filename, PETData.SinogramSize);
                case 'ACF'
                    filename  = [PETData.DataPath.rawdata_sino '\acf_00.a'];
                    data = read_sinograms(PETData,filename, PETData.SinogramSize);
                case 'ACF2'
                    filename  = [PETData.DataPath.rawdata_sino '\acf_second_00.a'];
                    data = read_sinograms(PETData,filename, PETData.SinogramSize);
                case 'scatters'
                    filename  = [PETData.DataPath.rawdata_sino '\scatter_estim2d_000000.s'];
                    scatter_2D = read_sinograms(PETData,filename, [PETData.SinogramSize(1:2) 127]);
                    scatter_3D = iSSRB(PETData,scatter_2D);
                    data = scatter_scaling(PETData,scatter_3D);
                    clear scatter_3D scatter_2D
            end
        end
        
        function data = read_32bit_listmode(PETData)
        end
        
        function data = read_sinograms(~,filename, SinoSize)
            fid=fopen(filename,'r');
            d = fread(fid,prod(SinoSize),'float');
            data = single(reshape(d,SinoSize));
            fclose(fid);
        end

        function Scatter3D = iSSRB(PETData,Scatter2D)
            
            nPlanePerSeg = Planes_Seg(PETData.Gantry);
            
            mo = cumsum(nPlanePerSeg)';
            no =[[1;mo(1:end-1)+1],mo];            

            Scatter3D = zeros(PETData.SinogramSize,'single');
            Scatter3D(:,:,no(1,1):no(1,2),:) = Scatter2D;
            
            for i = 2:2:length(nPlanePerSeg)
                
                delta = (nPlanePerSeg(1)- nPlanePerSeg(i))/2;
                indx = nPlanePerSeg(1) - delta;
                
                Scatter3D (:,:,no(i,1):no(i,2),:) = Scatter2D(:,:,delta+1:indx,:);
                Scatter3D (:,:,no(i+1,1):no(i+1,2),:) = Scatter2D(:,:,delta+1:indx,:);
            end
        end
        
        function scatter_3D = scatter_scaling(PETData,scatter_3D)
            P = PETData.Prompts;
            R = PETData.Randoms;
            NCF = PETData.NCF;
            ACF2 = PETData.ACF2;
            
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
            
            clear NCF ACF2 gaps R P Trues
        end
        
        function Check_interfile_list(PETData,FolderName)
            
        end
        

    end
    

    
    methods (Access = public)
        
        function  P = Prompts(PETData)
            P = Load_sinogram(PETData,'prompts');
        end
        function  R = Randoms(PETData)
            R = Load_sinogram(PETData,'randoms');
        end
        function  S = Scatters(PETData)
            S = Load_sinogram(PETData,'scatters');
        end
        function  ncf = NCF(PETData)
            ncf = Load_sinogram(PETData,'NCF');
        end
        function  acf = ACF(PETData)
            acf = Load_sinogram(PETData,'ACF');
        end
        
        function mu = mumap(PETData)
            mu = Load_image();
        end

        function info = get_acquisition_info(PETData)
            % get_acquisition_info() returns a structure contaning all key
            % infromation from dicom or interfile headers such injected dose,
            % acquisition time,...
            if (PETData.DataType == DataTypeEnum.sinogram_interfile)
                info = getInfoFromInterfile(PETData.DataPath.emission);
            elseif ( DataTypeEnum.list_mode_interfile)
                info = getInfoFromInterfile(PETData.DataPath.emission_listmode);
            end
        end
        
        function ListModeChopper(PETData)
            % ListModeChopper() splits list-mode data into a number of
            % specified frames and returns sinograms, based on MethodListData
            % it uses 'e7' HistogramReplay or 'matlab' read_32bit_listmode()
            % and histogram_data()
            PETData.histogram_data()
        end
        
        % Initalizes tima frames
        function InitFramesConfig(PETData, timeFrame_sec)
            % Get info from the header:
            info = getInfoFromInterfile(PETData.DataPath.emission_listmode);
            % Scan time:
            scanTime_sec = info.ImageDurationSec;
            % Frame durations (it has NumberOfFrames elements):
            PETData.FrameDuration_sec = timeFrame_sec;
            % Read from the header of the list file, the total time:
            % Number of frames.
            PETData.NumberOfFrames = scanTime_sec./timeFrame_sec;

            % List with the frames time stamps (it has NumberOfFrames elements):
            PETData.DynamicFrames_sec = zeros(PETData.NumberOfFrames,1);
        
            % Generate the time stamps for the frames:
            PETData.DynamicFrames_sec = 0 : timeFrame_sec : timeFrame_sec*PETData.NumberOfFrames;
        end
        
        function Dicom2IF()
            % converts Dicom to interfile, uses the interfile header
            % availabe in Dicom Private_0029_1010 field
        end
        
        function make_mu_map(),end
        
        function load_hardware_mu_map(), end
        
        function Members(PETData)
            %displays members and defaults values

        end
        
        function SinogramDisplay(PETData,sType)
            if strcmpi(sType,'ACF')
                temp = ACF(PETData);
            else
                temp = Prompts(PETData);
            end
            for i = 1:5:PETData.SinogramSize(2)
                drawnow,imshow(squeeze(temp(:,i,:)),[])
            end
            clear temp
        end
        
        function data = AxialCompress(PETData,span)
            sinogram_in = PETData.sinogram_size;
            PETData.span = span;
            % sinogram_out = init_sinogram_size(PETData, PETData.span, PETData.Gantry.nCrystalRings, PETData.Gantry.maxRingDiff);
            % Compress the emission sinogram:
            [sinogram_out, sinogram_size_out] = change_sinogram_span(PETData, sinogram, sinogram_size_in);
        end
        
        function PlotMichelogram(PETData)
            span = PETData.Gantry.span;
            maxRingDiff = PETData.Gantry.maxRingDiff;
            nCrystalRings  = PETData.Gantry.nCrystalRings;
            
            a = (span+1)/2;
            b = floor((maxRingDiff +1 - a)/span)*span ;
            c = maxRingDiff +1 - (a+b);
            
            nSeg = 2*floor((maxRingDiff - a)/span)+ 3;
            
            maxRingDiff = (maxRingDiff -c):-span:-(maxRingDiff -c);
            minRingDiff = (maxRingDiff -c -span+1):-span:-(maxRingDiff -c);
            if c~=0
                maxRingDiff = [ maxRingDiff, maxRingDiff,-(maxRingDiff-(c-1))];
                minRingDiff = [maxRingDiff-(c-1), minRingDiff, -maxRingDiff ];
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
            title(sprintf('No. Segments:%d, span: %d, maxRingDiff: %d', nSeg, span, maxRingDiff))
        end
        
        function uncompress(PETData,Dir)
            if nargin==1
                [name,pathstr]  = uigetfile('*.hdr');
                Dir = [pathstr name];
            end
            [pathstr,name] = fileparts(Dir);
            PETData.DataPath.emission = Dir;
            PETData.DataPath.emission_uncomp = [pathstr '\' name(1:end-2) '_uncomp.s.hdr'];
            uncompress_emission(PETData);
        end
    end
    
end








