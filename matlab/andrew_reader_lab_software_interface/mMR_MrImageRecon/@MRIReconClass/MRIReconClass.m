classdef MRIReconClass <handle
    
    properties (SetAccess = public)
        % Currently supports fully sampled data, thus it's assumed image matrix
        % size is the same as k-space size (nkSamples)
        isSimulation
        is3D
        imageType
        mri_DataPath
        mri_DicomPath
        PositionInfo
        kSpace
        kScaleFactor
        nkSamples
        nkSamplesRaw
        isUnderSampled
        kSpaceUnderSampled
        PhaseUnderSamplingFactor
        SliceUnderSamplingFactor
        CentralizingMask
        UnderSamplingMask
        nCoils
        coilRadious
        coilDistance
        nNavs
        kCentre
        PhaseFoV
        ReadFoV
        dThickness
        ImgSize
        VoxelSize
        ReconImgSize
        CoilSensitivityMap
        CoilEstimationMethod
        Walsh_CSM_smoothingFactor
        CoilSupportMask
        twixData
        Prior
        FovReductionFactor
    end
    
    methods
        function ObjMRI = MRIReconClass(varargin)
            % call the constructor
            ObjMRI.isSimulation     = 0;
            ObjMRI.imageType       = 'T1';
            ObjMRI.is3D             = [];
            ObjMRI.mri_DataPath     = '';
            ObjMRI.mri_DicomPath     = '';
            ObjMRI.PositionInfo     = [];
            ObjMRI.nCoils           = 5;       % mMR H&N coil
            ObjMRI.kSpace           = [];
            ObjMRI.kScaleFactor     = 1e10;
            ObjMRI.nkSamples        = [];
            ObjMRI.nkSamplesRaw     = [];
            ObjMRI.nNavs            = [];
            ObjMRI.kCentre          = [];
            ObjMRI.PhaseFoV         = 236.2500; % mMR H&N coil
            ObjMRI.ReadFoV          = 270;      % mMR H&N coil
            ObjMRI.dThickness       = 193.6;
            ObjMRI.coilRadious      = 100;
            ObjMRI.coilDistance     = 150;
            ObjMRI.ReconImgSize     = [];
            ObjMRI.isUnderSampled   = 0;
            ObjMRI.kSpaceUnderSampled           = [];
            ObjMRI.PhaseUnderSamplingFactor     = 1;
            ObjMRI.SliceUnderSamplingFactor     = 1;
            ObjMRI.CentralizingMask             = 1;
            ObjMRI.UnderSamplingMask            = 1;
            ObjMRI.CoilSensitivityMap           = [];
            ObjMRI.CoilEstimationMethod         = 'Walsh';
            ObjMRI.Walsh_CSM_smoothingFactor    = 40;
            ObjMRI.CoilSupportMask              = [];
            ObjMRI.ImgSize          = [];
            ObjMRI.VoxelSize        = [];
            ObjMRI.FovReductionFactor = [4,4,0]; % This is for realistic simulations, with an MR phantom that has the same FOV as PET.
            if isempty(varargin)
                [FileName,PathName] = uigetfile('*.dat');
                ObjMRI.mri_DataPath  = [PathName FileName];
            elseif isstruct(varargin{1})
                % update object's properties from user's options
                ObjMRI = getFiledsFromUsersOpt(ObjMRI,varargin{1});
                
                if ~ObjMRI.isSimulation && ~ObjMRI.isDat(ObjMRI.mri_DataPath),
                    error('TWIX data (.dat) should be specified');
                end
            end
            if ObjMRI.isSimulation
                % load phantom
                Ph = load_phantom(ObjMRI,varargin);
                
                %simulate data
                MRI_Simulation(ObjMRI,Ph);
                
            else
                
                read_raw(ObjMRI);
                CentralizekSpace(ObjMRI);
                if isempty(ObjMRI.CoilSensitivityMap)
                    fprintf('Coil sensitivty map estimation\n');
                    EstimateCoilSensitivityMap(ObjMRI);
                end
                
            end
            if ObjMRI.PhaseUnderSamplingFactor> 1 || ObjMRI.SliceUnderSamplingFactor > 1
                fprintf('Retrospective K-space undersampling\n');
                underSamplePhaseOrSliceEndoing(ObjMRI);
            end
            if ~isempty(ObjMRI.mri_DicomPath)
                ObjMRI.PositionInfo = getPatientPositionOrientation(ObjMRI.mri_DicomPath);
            else
                fprintf('MRI Dicom images should be specifid\n')
            end
            % update superclass properties
            p.ImageSize = ObjMRI.ImgSize;
            if ~ObjMRI.isSimulation
                p.imCropFactor = [5,7,7]; % for Siemsne mMR
            else
                p.imCropFactor = [5,5,0];
            end
            p.sWindowSize = 5;
            p.lWindowSize = 1;
            if isfield(varargin{1},'imCropFactor'), p.imCropFactor = varargin{1}.imCropFactor; end
            if isfield(varargin{1},'sWindowSize'), p.sWindowSize = varargin{1}.sWindowSize; end
            if isfield(varargin{1},'lWindowSize'), p.lWindowSize = varargin{1}.lWindowSize; end
            ObjMRI.Prior = PriorsClass(p);
            
        end
    end
    
    methods (Access = private)
        
        function ObjMRI = CentralizekSpace(ObjMRI)
            % For Siemens, the central frequncies are not at the center, so the data should be shifted
            delta = (ObjMRI.nkSamples/2+1) - ObjMRI.kCentre ;
            tempKspace = zeros([ObjMRI.nkSamples ObjMRI.nCoils],'single');
            
            ObjMRI.CentralizingMask = false(ObjMRI.nkSamples);
            
            for i = 1:ObjMRI.nCoils
                tempKspace(delta(1)+1:ObjMRI.nkSamples(1),delta(2)+1:ObjMRI.nkSamples(2),delta(3)+1:ObjMRI.nkSamples(3),i) = ObjMRI.kSpace(:,:,:,i);
                ObjMRI.CentralizingMask(delta(1)+1:ObjMRI.nkSamples(1),delta(2)+1:ObjMRI.nkSamples(2),delta(3)+1:ObjMRI.nkSamples(3)) = 1;
            end
            ObjMRI.kSpace = tempKspace;
            ObjMRI.UnderSamplingMask = ObjMRI.CentralizingMask;
        end
        
        function ObjMRI = underSamplePhaseOrSliceEndoing(ObjMRI)
            % undersample and zero-fill
            ObjMRI.isUnderSampled = ObjMRI.PhaseUnderSamplingFactor >1 || ObjMRI.SliceUnderSamplingFactor>1;
            
            ObjMRI.kSpaceUnderSampled = zeros([ObjMRI.nkSamples ObjMRI.nCoils],'single');
            ObjMRI.UnderSamplingMask = false(ObjMRI.nkSamples);
            
            for i = 1:ObjMRI.nCoils
                if ObjMRI.is3D
                    ObjMRI.kSpaceUnderSampled(:,1:ObjMRI.PhaseUnderSamplingFactor:ObjMRI.nkSamples(2),1:ObjMRI.SliceUnderSamplingFactor:ObjMRI.nkSamples(3),i) ...
                        = ObjMRI.kSpace(:,1:ObjMRI.PhaseUnderSamplingFactor:ObjMRI.nkSamples(2),1:ObjMRI.SliceUnderSamplingFactor:ObjMRI.nkSamples(3),i);
                    ObjMRI.UnderSamplingMask(:,1:ObjMRI.PhaseUnderSamplingFactor:ObjMRI.nkSamples(2),1:ObjMRI.SliceUnderSamplingFactor:ObjMRI.nkSamples(3)) =1;
                else
                    ObjMRI.kSpaceUnderSampled(:,1:ObjMRI.PhaseUnderSamplingFactor:ObjMRI.nkSamples(2),i) ...
                        = ObjMRI.kSpace(:,1:ObjMRI.PhaseUnderSamplingFactor:ObjMRI.nkSamples(2),i);
                    ObjMRI.UnderSamplingMask(:,1:ObjMRI.PhaseUnderSamplingFactor:ObjMRI.nkSamples(2)) =1;
                end
                
            end
            ObjMRI.UnderSamplingMask = logical(ObjMRI.UnderSamplingMask.*ObjMRI.CentralizingMask);
            
        end
        
        function scaledData = kScale(ObjMRI,data)
            ObjMRI.kScaleFactor  = ObjMRI.kScaleFactor ./(max(abs(data(:))));
            scaledData = ObjMRI.kScaleFactor * data;
        end
        
        function MRI_Simulation(ObjMRI,Ph)
            
            ObjMRI.ImgSize          = size(Ph);
            ObjMRI.is3D             = ndims(Ph)==3;
            ObjMRI.nkSamples        = ObjMRI.ImgSize;
            ObjMRI.CoilSupportMask              = ObjMRI.MakeCoilSupportMask(Ph);
            ObjMRI.CoilSensitivityMap           = ObjMRI.SimulateCoilSensitivityMap;
            ObjMRI.CentralizingMask             = true(ObjMRI.nkSamples);
            ObjMRI.UnderSamplingMask            = true(ObjMRI.nkSamples);
            ObjMRI.kSpace           = ObjMRI.kScale(ObjMRI.AddkSpaceNoise(ObjMRI.F(Ph,0))); %
%             ObjMRI.ReconImgSize     = ObjMRI.ImgSize;
        end
        
        function Ph = load_phantom(ObjMRI,varargin)
            if (size(varargin{1},2) >=2) && (~isempty(varargin{1}{2}))
                Ph = varargin{1}{2};
            elseif ~isempty(ObjMRI.mri_DataPath)
                Ph = load(ObjMRI.mri_DataPath,ObjMRI.imageType);
                Ph = Ph.(ObjMRI.imageType);
            else
                error('either provide image path or phantom')
            end
            
            ObjMRI.ReconImgSize = size(Ph);
            % call reduce FOV#
            Ph = ObjMRI.reduceFov(Ph);
        end
    end
    
    methods (Access = public)
        
        function twix_obj_fs = read_raw(ObjMRI)
            if isempty(ObjMRI.twixData)
                twix_obj_fs = mapVBVD(ObjMRI.mri_DataPath);
                ObjMRI.twixData = twix_obj_fs;
            end
            rawdata_fs  = ObjMRI.twixData.image.unsorted();
            
            % Measurement parameters
            ObjMRI.nCoils = ObjMRI.twixData.image.NCha;
            nAcq   = ObjMRI.twixData.image.NAcq; % Number of readouts acquired
            ObjMRI.nNavs  = ObjMRI.twixData.image.NSeg; % Number of cardiac cycles = Number of 2D iNavs
            
            nKx  = ObjMRI.twixData.image.NCol;
            nKy  = ObjMRI.twixData.image.NLin;
            nKz  = ObjMRI.twixData.image.NPar;
            
            centreKx = ObjMRI.twixData.image.centerCol(1);
            centreKy = ObjMRI.twixData.image.centerLin(1);
            centreKz = ObjMRI.twixData.image.centerPar(1);
            
            kY   = double(ObjMRI.twixData.image.Lin);
            kZ   = double(ObjMRI.twixData.image.Par);
            
            kSpaceData = zeros(nKx,nKy,nKz,ObjMRI.nCoils);
            
            %             kSamp  = zeros(nKx,nKy,nKz);
            for bbb=1:nAcq
                for c=1:ObjMRI.nCoils
                    kSpaceData(:,kY(bbb),kZ(bbb),c)=rawdata_fs(:,c,bbb);
                end
                %                 kSamp(:,kY(bbb),kZ(bbb)) = 1;
            end
            
            % update ObjMRI's properties
            ObjMRI.kSpace  = ObjMRI.kScale(kSpaceData);
            ObjMRI.is3D    = (nKz -1)>0;
            ObjMRI.nkSamples = [ObjMRI.twixData.hdr.Config.NColMeas, ObjMRI.twixData.hdr.Config.NLinMeas, ObjMRI.twixData.hdr.Config.NParMeas];
            ObjMRI.nkSamplesRaw = [nKx,nKy,nKz];
            ObjMRI.kCentre   = [centreKx centreKy centreKz];
            
            
            ObjMRI.PhaseFoV = ObjMRI.twixData.hdr.Meas.PhaseFoV;
            ObjMRI.ReadFoV  = ObjMRI.twixData.hdr.Meas.ReadFoV;
            ObjMRI.dThickness = ObjMRI.twixData.hdr.MeasYaps.sSliceArray.asSlice{1, 1}.dThickness;
            ObjMRI.ReconImgSize = [ObjMRI.twixData.hdr.Config.NImageCols,...
                ObjMRI.twixData.hdr.Config.NImageLins,...
                ObjMRI.twixData.hdr.Config.NImagePar];
            
            ObjMRI.ImgSize = ObjMRI.nkSamples;
            
            ObjMRI.VoxelSize = [ObjMRI.ReadFoV./ObjMRI.ReconImgSize(1), ObjMRI.PhaseFoV./ObjMRI.ReconImgSize(2),...
                ObjMRI.dThickness./ObjMRI.ReconImgSize(3)];
            
        end
        
        function x = isDat(~,inputFile)
            [~,~,ext] = fileparts(inputFile) ;
            if strcmpi(ext,'.dat')
                x = 1;
            else
                x = 0;
            end
        end
        
        function ObjMRI = Revise(ObjMRI,opt)
            % to revise the properties of a given object without
            % re-instantiation
            vfields = fieldnames(opt);
            prop = properties(ObjMRI);
            for i = 1:length(vfields)
                field = vfields{i};
                if sum(strcmpi(prop, field )) > 0
                    ObjMRI.(field) = opt.(field);
                end
            end
            
            if isfield(opt,'PhaseUnderSamplingFactor') || isfield(opt,'SliceUnderSamplingFactor')
                underSamplePhaseOrSliceEndoing(ObjMRI);
            end
            if isfield(opt,'sWindowSize') || isfield(opt,'lWindowSize') ...
                    || isfield(opt,'imCropFactor') || isfield(opt,'ImageSize')
                ObjMRI.Prior = PriorsClass(opt);
            end
            
            
        end
        
        function m = F0(ObjMRI,x,mask)
            if ObjMRI.isSimulation
                m = fftn(x).*mask;
            else
                m = ifftshift(ifftshift(ifftshift(fftn(fftshift(fftshift(fftshift(x,1),2),3)),1),2),3).*mask;%
            end
        end
        
        function x = FH0(ObjMRI,m,mask)
            if ObjMRI.isSimulation
                x = ifftn(m.*mask);
            else
                x = fftshift(fftshift(fftshift(ifftn(ifftshift(ifftshift(ifftshift(m.*mask,1),2),3)),1),2),3);
            end
        end
        
        function m = F(ObjMRI,x,underSampling)
            % perfroms MRI forward operator
            if nargin==2, underSampling = 1; end
            if underSampling
                mask = ObjMRI.UnderSamplingMask;
            else
                mask = ObjMRI.CentralizingMask;
            end
            
            m = squeeze(zeros([ObjMRI.nkSamples, ObjMRI.nCoils],'single'));
            
            for i = 1:ObjMRI.nCoils
                if isempty(ObjMRI.CoilSensitivityMap)
                    Sen = 1;
                else
                    if ObjMRI.is3D
                        Sen = ObjMRI.CoilSensitivityMap(:,:,:,i);
                    else
                        Sen = ObjMRI.CoilSensitivityMap(:,:,i);
                    end
                end
                
                K = ObjMRI.F0(x.*Sen,mask);
                if ObjMRI.is3D
                    m(:,:,:,i) = K;
                else
                    m(:,:,i) = K;
                end
                
            end
        end
        
        function x = FH(ObjMRI,m,underSampling)
            % perfroms MRI adjoint operator
            if nargin==2, underSampling = 1; end
            if underSampling
                mask = ObjMRI.UnderSamplingMask;
            else
                mask = ObjMRI.CentralizingMask;
            end
            x = zeros(ObjMRI.nkSamples);
            for i = 1: ObjMRI.nCoils
                if isempty(ObjMRI.CoilSensitivityMap)
                    Sen = 1;
                else
                    if ObjMRI.is3D
                        Sen = ObjMRI.CoilSensitivityMap(:,:,:,i);
                        K = m(:,:,:,i);
                    else
                        Sen = ObjMRI.CoilSensitivityMap(:,:,i);
                        K = m(:,:,i);
                    end
                end
                
                x = x + conj(Sen).*ObjMRI.FH0(K,mask);
            end
        end
        
        function x = FHF(ObjMRI,m,underSampling)
            % perfroms MRI Gram Matrix opertor
            if nargin==2, underSampling = 1; end
            
            x = ObjMRI.FH(ObjMRI.F(m,underSampling),underSampling);
        end
        
        function Y = RSS(ObjMRI,X)
            % Calculates root of sum of squares
            if ndims(X)==2 %#ok<ISMAT> % Gradient vectors
                dims = 2;
            else % images
                if ObjMRI.is3D
                    dims = 4;
                else
                    dims = 3;
                end
            end
            Y  = sqrt(sum(abs(X).^2,dims));
        end
        
        function [ImgCoils, ImgCoilsSOS] = RecCoilsImg(ObjMRI)
            % ImgCoils from fully-sampled k-space, so use CentralizingMask
            ImgCoils = zeros([ObjMRI.nkSamples, ObjMRI.nCoils],'single');
            for i = 1:ObjMRI.nCoils
                if ObjMRI.is3D
                    ImgCoils(:,:,:,i) = ObjMRI.FH0(ObjMRI.kSpace(:,:,:,i),ObjMRI.CentralizingMask);
                else
                    ImgCoils(:,:,i) = ObjMRI.FH0(ObjMRI.kSpace(:,:,i),ObjMRI.CentralizingMask);
                end
                
            end
            
            ImgCoilsSOS = ObjMRI.RSS(ImgCoils);
            ObjMRI.CoilSupportMask = ObjMRI.MakeCoilSupportMask(ImgCoilsSOS);
        end
        
        function mask = MakeCoilSupportMask(ObjMRI,imgCoilSOS)
            
            if ObjMRI.is3D
                mask = imgaussfilt3(imgCoilSOS./max(imgCoilSOS(:)),8);
            else
                mask = imgaussfilt(imgCoilSOS./max(imgCoilSOS(:)),8);
            end
            
            level = graythresh(mask);
            if level==0
                fprintf('MakeCoilSupportMask:: zero threshold level !!\n')
                mask = 1;
                return
            end
            mask = mask >(level*0.25);
            [x,y,z] = meshgrid(-1:1:1);
            r = (x.^2+y.^2+z.^2)<2;
            
            for i = 1:10
                mask = imdilate(single(mask),r);
            end
            if ObjMRI.is3D
                for i=1:size(mask,3)
                    mask(:,:,i) = imfill(mask(:,:,i),'holes');
                end
            else
                mask= imfill(mask,'holes');
            end
            mask = logical(mask);
        end
        
        function EstimateCoilSensitivityMap(ObjMRI)
            % currently is3D = 1
            [ImgCoils, ImgCoilsSOS] = RecCoilsImg(ObjMRI);
            if strcmpi(ObjMRI.CoilEstimationMethod,'Walsh')
                
                smoothing = ObjMRI.Walsh_CSM_smoothingFactor;
                chunks = ObjMRI.nkSamples(3);
                CoilEst = ismrm_estimate_csm_walsh_3D(ImgCoils, smoothing, chunks);
                
            else
                strcmpi(ObjMRI.CoilEstimationMethod,'rsos')
                
                CoilEst = ImgCoils./repmat(ImgCoilsSOS,[1,1,1,3]);
            end
            
            % mask the CSMs
            for i = 1:ObjMRI.nCoils
                CoilEst(:,:,:,i) = CoilEst(:,:,:,i).*ObjMRI.CoilSupportMask;
            end
            ObjMRI.CoilSensitivityMap = single(CoilEst);
        end
        
        function S = SimulateCoilSensitivityMap(ObjMRI,coilRadious,coilDistance)
            
            % Generates a stack of 2D sensitivity maps that are simulated with the
            % Biot-Savart law. The coils are cicular, with centers which are
            % equidistant to the origin, and their axis are uniformly distributed radiuses.
            %
            %  Matthieu Guerquin-Kern, 2012
            
            FOV = ObjMRI.PhaseFoV/1000*[1,1]; % meters
            res = FOV./ObjMRI.ImgSize(1:2);
            Nc = ObjMRI.nCoils;
            if nargin==1
                R = ObjMRI.coilRadious/1000;
                D = ObjMRI.coilDistance/1000;
            else
                R = coilRadious/1000;
                D = coilDistance/1000;
            end
            
            x = 0:res(1):FOV(1)-res(1);
            x = x-x(ceil((end+1)/2));
            
            y = 0:res(2):FOV(2)-res(2);
            y = y-y(ceil((end+1)/2));
            
            if numel(Nc)==1
                dalpha = 2*pi/Nc;
                alpha = 0:dalpha:2*pi-dalpha;
            else
                alpha = Nc;
                Nc = numel(alpha);
            end
            S = zeros(numel(x),numel(y),Nc);
            Nangles = 60;
            dtheta = 2*pi/Nangles;
            theta = -pi:dtheta:pi-dtheta;
            [Y,X,T] = ndgrid(x,y,theta);
            
            for i = 1:Nc
                x = X*cos(alpha(i))-Y*sin(alpha(i));
                y = X*sin(alpha(i))+Y*cos(alpha(i));
                s = exp(1i*alpha(i))*(-R+y.*cos(T)-1j*(D-x).*cos(T))./((D-x).^2+y.^2+R^2-2*R*y.*cos(T)).^(3/2);
                S(:,:,i) = dtheta*sum(s,3);
            end
            
            % % Normalization
            S = S/max(abs(S(:)));
            
            if ObjMRI.is3D
                S = repmat(S,[1,1,1,ObjMRI.ImgSize(3)]);
                S = permute(S,[1,2,4,3]);
            end
            
            % mask the CSMs
            for i = 1:ObjMRI.nCoils
                if ObjMRI.is3D
                    S(:,:,:,i) = S(:,:,:,i).*ObjMRI.CoilSupportMask;
                else
                    S(:,:,i) = S(:,:,i).*ObjMRI.CoilSupportMask;
                end
            end
        end
        
        function n = AddkSpaceNoise(ObjMRI,m)
            
            snr_hf = 0.1;
            m = reshape(m,prod(ObjMRI.nkSamples),ObjMRI.nCoils);
            n = randn(size(m))+1j*randn(size(m));
            n = n - repmat(mean(n,1),[size(m,1),1]);
            snr_hf = snr_hf*std(m(:))/std(n(:));
            fprintf('SNR: %f\n',20*log10(snr_hf));
            
            n = m + snr_hf *n ;
            n = reshape(n,[ObjMRI.nkSamples,ObjMRI.nCoils]);
        end
        
        function x = Magnitude(~,x)
            x = sqrt(sum(abs(x(:)).^2));
        end
        
        function Y = softThreshold(ObjMRI,Norm,Z,rho,lambda,w)
            
            Norm = repmat(Norm + 1e-5, [1,ObjMRI.Prior.nS] );
            Y =  max(0, Norm - w./(rho/lambda)).*(Z./Norm);
        end
        
        function Xi = Trajectory(ObjMRI,q)
            % Compute sampling pattern from kspace
            
            % stack of 2D radial trajectories
            % only for simulation of square matrix size
            
            % q number of stokes
            n = ObjMRI.nkSamples(1);
            Theta = linspace(0,pi,q+1);Theta(end) = [];
            Xi = zeros(n,n);
            for theta = Theta
                t = linspace(-1,1,3*n)*n;
                X = round(t.*cos(theta)) + n/2+1;
                Y = round(t.*sin(theta)) + n/2+1;
                I = find(X>0 & X<=n & Y>0 & Y<=n);
                X = X(I);
                Y = Y(I);
                Xi(X+(Y-1)*n) = 1;
            end
            
            Xi = fftshift(Xi);
            if ObjMRI.is3D
                Xi =repmat(Xi,[1,1,ObjMRI.nkSamples(3)]);
            end
        end
        %{

        
function ISMRM(objMRI)
            %Import ISMRM raw data files
        end
        
function Y = nufft(objMRI,X,options)
        end
        
function Y = Homodyne(objMRI, X, dim, fraction)
            %
            % 	Performs homodyne reconstruction along {dim}. Where {fraction}
            % 	describes how much of k-space has been acquired along this
            % 	dimension.
            
        end
        %}
        
        function display(ObjMRI)
            
            disp(ObjMRI)
            methods(ObjMRI)
            
        end
        
        function saveAt(~,image, fname)
            fid = fopen(fname,'w');
            fwrite(fid,image,'float');
            fclose(fid);
        end
        
        function Img = CropOrient(ObjMRI, Img)
            % into the dicom image space
            delta = (ObjMRI.ImgSize - ObjMRI.ReconImgSize)/2;
            Img = Img(delta(1)+1:ObjMRI.ImgSize(1)-delta(1),...
                delta(2)+1:ObjMRI.ImgSize(2)-delta(2),...
                delta(3)+1:ObjMRI.ImgSize(3)-delta(3));
            
            Img = flip(flip(Img,2),3);
            % shift by [0 1 1]% consider it in centrelizing k-space
            Img = circshift(Img,0,1);
            Img = circshift(Img,1,2);
            Img = circshift(Img,1,3);
        end
        
        function ImgOut = UndoCropOrient(ObjMRI,Img)
            % into mr native space
            Img = circshift(Img,0,1);
            Img = circshift(Img,-1,2);
            Img = circshift(Img,-1,3);
            
            Img = flip(flip(Img,3),2);
            
            delta = (ObjMRI.ImgSize - ObjMRI.ReconImgSize)/2;
            ImgOut = zeros(ObjMRI.ImgSize,'single');
            
            ImgOut(delta(1)+1:ObjMRI.ImgSize(1)-delta(1),...
                delta(2)+1:ObjMRI.ImgSize(2)-delta(2),...
                delta(3)+1:ObjMRI.ImgSize(3)-delta(3)) = Img;
        end
        
        function [Img, imageRef3d] = applyAffineTransfrom(ObjMRI,image)
            
            matlabAffine = affine3d(ObjMRI.PositionInfo.affineMatrix');
            
            inImageRef3d = imref3d(ObjMRI.PositionInfo.ImageSize, ObjMRI.PositionInfo.pixelSpacing_mm(1), ObjMRI.PositionInfo.pixelSpacing_mm(2), ObjMRI.PositionInfo.sliceThickness);
            
            [Img, imageRef3d] = imwarp(image,inImageRef3d, matlabAffine);
            imageRef3d.XWorldLimits = imageRef3d.XWorldLimits + ObjMRI.PositionInfo.posTopLeftPixel_1(1);
            
            imageRef3d.YWorldLimits = imageRef3d.YWorldLimits + ObjMRI.PositionInfo.posTopLeftPixel_1(2);
            
            imageRef3d.ZWorldLimits = imageRef3d.ZWorldLimits + ObjMRI.PositionInfo.posTopLeftPixel_1(3);
            
            
            if max(ObjMRI.PositionInfo.dirZ) > 0
                Img(:,:,1:end) = Img(:,:,end:-1:1);
            end
            %}
        end
        
        function [Img, refResampledImage] = applyInverseAffineTransfrom(ObjMRI,image, imageRef3d)
            
            if max(ObjMRI.PositionInfo.dirZ) > 0    % Use max because it can have serveral component if the image is rotated coronally:
                image(:,:,1:end) = image(:,:,end:-1:1);
            end
            
            matlabAffine = affine3d(ObjMRI.PositionInfo.affineMatrix');
            [image_TT,imageRefTT ] = imwarp(image,imageRef3d, invert(matlabAffine));
            
            inImageRef3d = imref3d(ObjMRI.PositionInfo.ImageSize, ObjMRI.PositionInfo.pixelSpacing_mm(1), ObjMRI.PositionInfo.pixelSpacing_mm(2), ObjMRI.PositionInfo.sliceThickness);
            
            imageRefTT2 = imref3d(imageRefTT.ImageSize, imageRefTT.PixelExtentInWorldX, imageRefTT.PixelExtentInWorldY, imageRefTT.PixelExtentInWorldZ);
            [Img, refResampledImage] = ImageResample(image_TT,imageRefTT2, inImageRef3d);
        end
        
        function [Img,ImgRef3d] = mapMrNativeSpaceToReferenceSpace(ObjMRI,image)
            if ObjMRI.isSimulation
                Img = image;
                ImgRef3d = [];
            else
                image = ObjMRI.CropOrient(image);
                [Img, ImgRef3d] = ObjMRI.applyAffineTransfrom(image);
                ObjMRI.PositionInfo.MrInRefSpace = ImgRef3d;
            end
        end
        
        function ImgOut = mapReferenceSpaceToMrNativeSpace(ObjMRI,Img,ImgRef3d)
            if ObjMRI.isSimulation
                ImgOut = Img;
            else
                if nargin==2
                    ImgRef3d = ObjMRI.PositionInfo.MrInRefSpace;
                end
                image = ObjMRI.applyInverseAffineTransfrom(Img,ImgRef3d);
                ImgOut = ObjMRI.UndoCropOrient(image);
            end
        end
        
        %         function get_PetRefCorrdinate
        function [Img,ImgRef3d,refResampledImage] = mapMrNativeSpaceToPETSpace(ObjMRI, image,petRef)
            if ObjMRI.isSimulation
                Img = ObjMRI.UndoReduceFov(image);
                ImgRef3d = [];
                refResampledImage = [];
            else
                % map mr native space to the scanner's reference and
                % down-sample to PET resolution
                if nargin==2
                    petRef = ObjMRI.PositionInfo.PetRef;
                end
                
                [Img,ImgRef3d] = ObjMRI.mapMrNativeSpaceToReferenceSpace(image);
                [Img, refResampledImage] = ImageResample(Img,ImgRef3d, petRef);
                ObjMRI.PositionInfo.MrInPetSpace = refResampledImage;
            end
        end
        
        function ImgOut = mapPetSpaceToMrNativeSpace(ObjMRI, image,refResampledImage,imageRef3d)
            if ObjMRI.isSimulation
                ImgOut = ObjMRI.reduceFov(image);
            else
                if nargin==2
                    imageRef3d = ObjMRI.PositionInfo.MrInRefSpace;
                    refResampledImage = ObjMRI.PositionInfo.MrInPetSpace;
                end
                Img = ImageResample(image,refResampledImage,imageRef3d );
                
                ImgOut = mapReferenceSpaceToMrNativeSpace(ObjMRI,Img,imageRef3d);
            end
        end
        
        function setPositionInfo(ObjMRI,path)
            ObjMRI.PositionInfo = getPatientPositionOrientation(path);
        end
        
        function setPetPositionInfo(ObjMRI,refPET)
            
            ObjMRI.PositionInfo.PetRef = refPET;
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function [Img,newSize] = reduceFov(ObjMRI,Img)
            if all(ObjMRI.FovReductionFactor==0)
                newSize = ObjMRI.ReconImgSize;
                if nargin==1
                    Img = [];
                end
            else
                if length(ObjMRI.FovReductionFactor)== 1
                    if ObjMRI.is3D
                        ObjMRI.FovReductionFactor = ObjMRI.FovReductionFactor*[1 1 0];
                    else
                        ObjMRI.FovReductionFactor = ObjMRI.FovReductionFactor*[1 1];
                    end
                end
                
                J = 0;
                if ObjMRI.FovReductionFactor(1)
                    ObjMRI.FovReductionFactor(1) = max(2.5, ObjMRI.FovReductionFactor(1));
                    J = floor(ObjMRI.ReconImgSize(1)/ObjMRI.FovReductionFactor(1));
                end
                
                I = 0;
                if ObjMRI.FovReductionFactor(2)
                    ObjMRI.FovReductionFactor(2) = max(2.5, ObjMRI.FovReductionFactor(2));
                    I = floor(ObjMRI.ReconImgSize(2)/ObjMRI.FovReductionFactor(2));
                end
                
                if ObjMRI.is3D
                    K = 0;
                    if ObjMRI.FovReductionFactor(3)
                        ObjMRI.FovReductionFactor(3) = max(2.5, ObjMRI.FovReductionFactor(3));
                        K = floor(ObjMRI.ReconImgSize(3)/ObjMRI.FovReductionFactor(3));
                    end
                    newSize = [length((J:(ObjMRI.ReconImgSize(1)-J-1))+1),length((I:(ObjMRI.ReconImgSize(2)-I-1))+1),length((K:(ObjMRI.ReconImgSize(3)-K-1))+1)];
                else
                    newSize = [length((J:(ObjMRI.ReconImgSize(1)-J-1))+1),length((I:(ObjMRI.ReconImgSize(2)-I-1))+1)];
                    if length(ObjMRI.ReconImgSize)==3
                        newSize = [ newSize ,1];
                    end
                end
                if nargin==1
                    Img = [];
                else
                    if ObjMRI.is3D
                        Img = Img((J:(ObjMRI.ReconImgSize(1)-J-1))+1,(I:(ObjMRI.ReconImgSize(2)-I-1))+1,(K:(ObjMRI.ReconImgSize(3)-K-1))+1);
                    else
                        Img = Img((J:(ObjMRI.ReconImgSize(1)-J-1))+1,(I:(ObjMRI.ReconImgSize(2)-I-1))+1);
                    end
                end
            end
        end
        
        function ImgNew = UndoReduceFov(ObjMRI,Img)
            if all(ObjMRI.FovReductionFactor==0)
                ImgNew = Img;
                return
            end
            ImgNew = zeros(ObjMRI.ReconImgSize,'single');
            
            S = (ObjMRI.ReconImgSize - ObjMRI.ImgSize)/2;
            J = S(1); I = S(2);
            if ObjMRI.is3D
                K = S(3);
                ImgNew((J:(ObjMRI.ReconImgSize(1)-S(1)-1))+1,(I:(ObjMRI.ReconImgSize(2)-I-1))+1,(K:(ObjMRI.ReconImgSize(3)-K-1))+1) = Img;
            else
                ImgNew((J:(ObjMRI.ReconImgSize(1)-S(1)-1))+1,(I:(ObjMRI.ReconImgSize(2)-I-1))+1) = Img;
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        
    end
end






