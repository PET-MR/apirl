classdef PETDataClass < handle
    properties (SetAccess = private)
        
        DataPath
        % The folder path containing raw data in Dicom IMA and PDT or Interfile
        % format. If Dicom, the folder should contain 1) emission data,
        % 2) normalization file and 3) human u-map. If Interfile, it should
        % contain 1) emission.s.hdr 2) norm.n.hdr, 3) umap_hardware.mhdr
        % and 4) umap_human.mhdr
        
        ImageSize
        %
        SinogramSize
        % [nRadialBin, nAngularBins,nSinogramPlanes]
        
        Gantry
        % A structure with fields: .Span, .MRD ,.nCrystalRings
        
        MethodSinoData %'e7', 'stir'.
        % Use e7 tools to generate all sinogram data or the combination of
        % STIR ans Apirl
        
        MethodListData % 'e7', 'matlab'
        % Use e7 tools to process list mode data or a matlab version (to be
        % developed
        
        DynamicFrames
        
        MotionFrames
        
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
            PETData.DataPath.emission           = '';
            PETData.DataPath.norm               = '';
            PETData.DataPath.umap               = '';
            PETData.DataPath.hardware_umap      = '';
            PETData.DataPath.rawdata_sino       = '';
            PETData.DataPath.scatters           = '';
            PETData.SinogramSize                = [344,252,837];
            PETData.Gantry.Span                 = 11;
            PETData.Gantry.MRD                  = 60;
            PETData.Gantry.nCrystalRings        = 64;
            PETData.MethodSinoData              = 'e7';
            PETData.MethodListData              = 'e7';
            PETData.DynamicFrames               = 1;
            PETData.MotionFrames                = 0;
            PETData.FrameDuration_sec               = 3600; %sec.
            PETData.SoftwarePaths.STIR         = '';
            PETData.SoftwarePaths.Apirl        = '';        
            if(strcmp(computer(), 'GLNXA64')) % If linux, call wine.
                PETData.SoftwarePaths.e7.siemens   = 'wine C:\Siemens\PET\bin.win64-VA20\e7_recon.exe';
                PETData.SoftwarePaths.e7.JSRecon12 = 'wine cscript C:\JSRecon12\JSRecon12.js ';
            else
                PETData.SoftwarePaths.e7.siemens   = 'C:\Siemens\PET\bin.win64-VA20\e7_recon.exe';
                PETData.SoftwarePaths.e7.JSRecon12 = 'cscript C:\JSRecon12\JSRecon12.js ';
            end
            
            % If not any path, ask for it.
            if nargin == 0
            % prompt a window to select the data path
                FolderName = uigetdir();
                PETData = PETDataClass(FolderName);
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
                    %Gantry.Span and Gantry.MRD

                elseif isdir(varargin{1})
                    FolderName = varargin{1};
                    [DataType] = Which(PETData,FolderName);
                    switch(DataType)
                        case isInterfile
                            % Read sinogram or listmode file in interfile format and 
                            % return PETData.DataPath structure
                            PETData = read_check_interfiles(PETData, FolderName);
                        case isCompressedSino(FolderName)
                            uncompress_emission(PETData);
                        case dicom
                            %  Call JSRecon12 to generate Siemens interfiles and
                            %  return PETData.DataPath structure
                            fprintf('Calling JSRecon12...\n');
                            PETData = prompt_JSRecon12(PETData, FolderName);
                        case list_mode
                            PETData = histogram_data(PETData, lists, SinogramSize);
                    end
                else

                end
            end
        end
    end
    
    
    methods (Access = private)
        
        PETData = prompt_JSRecon12(PETData, FolderName);
        PETData = read_check_interfiles(PETData, FolderName);

        function [DicomFile,IntFile] = Which(~,path)

            listing = dir(path);
            for i = 3:length(listing)
                [~, ~, ext] = fileparts(listing(i).name);
                if strcmpi(ext,'.IMA') || strcmpi(ext,'.PDT')
                    DataType = DataTypeEnum.dicom;
                else
                    if strcmpi(ext,'.hdr') || strcmpi(ext,'.mhdr')
                        [dummy,name,ext] = fileparts(name);
                        if strcmpi(ext,'.s') 
                            DataType = DataTypeEnum.interfile;
                        elseif strcmpi(ext,'.l') 
                            DataType = DataTypeEnum.list_mode;
                        end
                    end
                end
            end
            if ~DicomFile && ~IntFile
                error('No Dicom/Interfile was found');
            end
            if DicomFile && IntFile
                error('Please seperate the interfile and dicom files')
            end
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
                if strcmpi(PETData.MethodSinoData,'e7') && (PETData.Gantry.Span ==11)
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
        
        function data = histogram_data(PETData, lists, SinogramSize)
            % calls matlab's accumarray to generate prompts/delays
            % sinograms from the provided lists
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

        function info = get_acquisition_info()
            % get_acquisition_info() returns a structure contaning all key
            % infromation from dicom or interfile headers such injected dose,
            % acquisition time,...
        end
        
        function ListModeChopper()
            % ListModeChopper() splits list-mode data into a number of
            % specified frames and returns sinograms, based on MethodListData
            % it uses 'e7' HistogramReplay or 'matlab' read_32bit_listmode()
            % and histogram_data()
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
        
        function data = AxialCompress(PETData,data)
        end
        
        function PlotMichelogram(PETData)
            Span = PETData.Gantry.Span;
            MRD = PETData.Gantry.MRD;
            nCrystalRings  = PETData.Gantry.nCrystalRings;
            
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








