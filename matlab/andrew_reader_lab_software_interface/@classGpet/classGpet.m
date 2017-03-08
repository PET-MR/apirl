% *********************************************************************
% Reconstruction Framework for Siemens Biograph mMR.  
% class: Gpet
% Authors: Martin Belzunce, Abolfazl Mehranian. Kings College London.
% Date: 08/02/2016
% *********************************************************************
% Definition of the class classGpet to manage projector/backprojectors and
% other tools used in the image reconstruction of Siemens Biograph mMR
% data.

classdef classGpet < handle
    properties (SetAccess = private)
        % Type of scanner. Options: '2D_radon', 'mMR', 'cylindrical'
        scanner 
        % Properties of the scanner. This is an structure with additional
        % parameters specific for each scanner, for example for cylindrical
        % scanner it has the radius_mm propery.
        scanner_properties
        % Sinogram size:
        sinogram_size     % Struct with the size of the sinogram, it can be 2d, multislice2d and 3d.
        % Image size:
        image_size
        % Bed position (introduces an offset in the axial axis):
        bed_position_mm
        % Ref structure:
        ref_image
        % Projector/Backrpojector. Options:
        % 'pre-computed_matlab','otf_matlab', 'otf_siddon_cpu','otf_siddon_gpu'
        method
        % PSF. Oprions: 'shift-invar', 'shift-var', 'none'
        PSF
        % Number of iterations:
        nIter
        % Number of subsets:
        nSubsets
        % Radial Bin trimming
        radialBinTrim
        % Temporary files path:
        tempPath
        % Flag to delete the temp folder when we're finished (SAM ELLIS,
        % 24/08/2016)
        deleteTemp
        % Asymmertic pre-computed sysetm matrix
        Geom
        % Method to estimate randoms:
        method_for_randoms
        % Method to estimate scatter:
        method_for_scatter
        % Method for normalization:
        method_for_normalization    % 'cbn_expansion', 'from_e7_binary_interfile'
        % Operative system.
        os
        % bar for the paths.
        bar
        % number of Siddon rays per LOR (Transverse direction)
        nRays
        % number of Siddon rays per LOR in the axial direction
        % (axialDirection)
        nAxialRays
        % Verbosity level: 0, 1, 2. (0 silent)
        verbosity
    end
    
    methods
        % Constructors:
        function objGpet = classGpet(varargin) % Default options: (). From file (filename)
            if(strcmp(computer(), 'GLNXA64')) % If linux, call wine64.
                objGpet.os = 'linux';
                objGpet.bar = '/';
            else
                objGpet.os = 'windows';
                objGpet.bar = '\';
            end
            
            if (nargin == 0)
                objGpet.scanner = 'mMR';
            else
                if isfield(varargin{1},'scanner')
                    objGpet.scanner = varargin{1}.scanner;
                elseif length(varargin)>=2 && ischar(varargin{1}) % pair name,values
                    i = find(cellfun(@any,strfind(varargin,'scanner')));
                    objGpet.scanner = varargin{i+1};
                else
                    objGpet.scanner = 'mMR';
                end
            end
            objGpet.initScanner();
            objGpet.nRays = 1;
            objGpet.nAxialRays = 1;
            objGpet.method =  'otf_siddon_cpu';
            objGpet.PSF.type = 'shift-invar';
            objGpet.PSF.Width = 4; %mm
            objGpet.radialBinTrim = 0;
            objGpet.Geom = '';
            % SAM ELLIS EDIT (25/08/2016): changed the '.' to pwd so that
            % when deleting temp files, don't need to be in the original
            % directory
            objGpet.tempPath = [pwd objGpet.bar 'temp' objGpet.bar];
            objGpet.deleteTemp = false; % SAM ELLIS 24/08/2016
            objGpet.verbosity = 0;
            objGpet.method_for_randoms = 'from_ML_singles_matlab';
            objGpet.method_for_normalization = 'cbn_expansion';
            if nargin == 1
                % Read configuration from file or from struct:
                if isstruct(varargin{1})
                    if ~isfield(varargin{1}, 'scanner') || ~isfield(varargin{1}, 'method')
                        disp('Configuration for''otf_siddon_cpu''');
                    end
                    Revise(objGpet,varargin{1});
                    
                % Read configuration from file:
                elseif ischar(varargin{1})
                    objGpet.readConfigFromFile(varargin{1})
                end
            else
                % Read configuration from name/value pairs:
                objGpet = varargin_pair(objGpet, varargin);
            end
            if ~isdir(objGpet.tempPath)
                mkdir(objGpet.tempPath)
            end
            
            if (objGpet.sinogram_size.span ~=11) || (objGpet.sinogram_size.maxRingDifference ~=60 ) % default values
                init_sinogram_size(objGpet, objGpet.sinogram_size.span, objGpet.sinogram_size.nRings, objGpet.sinogram_size.maxRingDifference);
            end
            if objGpet.sinogram_size.span == 0
                objGpet.image_size.voxelSize_mm(3) = objGpet.image_size.voxelSize_mm(3)*objGpet.image_size.matrixSize(3)/objGpet.sinogram_size.nRings;
                objGpet.image_size.matrixSize(3)=objGpet.sinogram_size.nRings;
                warning('Overriding the image size to the number of rings to work in multislice 2d. Now the image size is %dx%dx%d and voxel size is %fx%fx%f.', ...
                    objGpet.image_size.matrixSize(1), objGpet.image_size.matrixSize(2), objGpet.image_size.matrixSize(3),...
                    objGpet.image_size.voxelSize_mm(1), objGpet.image_size.voxelSize_mm(2), objGpet.image_size.voxelSize_mm(3));
            elseif objGpet.sinogram_size.span == -1
                % Keep the voxel sizeobjGpet.image_size.voxelSize_mm(3)
                objGpet.image_size.matrixSize(3)=objGpet.sinogram_size.nRings;
                
                % SAM ELLIS EDIT: removed this warning, since span = -1
                % gives nRings = 1 as desired, so no warning necessary
%                 warning('Overriding the image size to the number of rings to work in multislice 2d. Now the image size is %dx%dx%d and voxel size is %fx%fx%f.', ...
%                     objGpet.image_size.matrixSize(1), objGpet.image_size.matrixSize(2), objGpet.image_size.matrixSize(3),...
%                     objGpet.image_size.voxelSize_mm(1), objGpet.image_size.voxelSize_mm(2), objGpet.image_size.voxelSize_mm(3));
            end
            objGpet.bed_position_mm = 0;
            objGpet.init_ref_image();
        end
        
        function objGpet = readConfigFromFile(objGpet, strFilename)
            error('todo: read configuration from file');
        end
        
        % Function that intializes the scanner:
        function initScanner(objGpet)
            if strcmpi(objGpet.scanner,'2D_radon')
                disp('Configuration for ''2D_radon'' scanner');
                objGpet.G_2D_radon_setup();
            elseif strcmpi(objGpet.scanner,'mMR')
                disp('Configuration for ''mMR'' scanner');
                objGpet.G_mMR_setup();
            elseif strcmpi(objGpet.scanner,'cylindrical')
                disp('Configuration for ''cylindrical'' scanner');
                objGpet.G_cylindrical_setup();
            else
                error('unkown scanner')
            end
        end
        


        function G_2D_radon_setup(objGpet)
            % Default parameter, only if it havent been loaded by the
            % config previously:
            if isempty(objGpet.image_size)
                objGpet.image_size.matrixSize = [512, 512, 1];
                objGpet.image_size.voxelSize_mm = [1, 1, 1];
            end
            if isempty(objGpet.sinogram_size)
                x = radon(ones(objGpet.image_size.matrixSize(1:2)),0:179);
                x = size(x);
                objGpet.sinogram_size.nRadialBins = x(1);
                objGpet.sinogram_size.nAnglesBins = x(2);
                objGpet.sinogram_size.nSinogramPlanes = 1;
                % EDIT: SAM ELLIS - 23/08/2016
                % NEED TO ADD OTHER DEFAULTS FOR 2D RADON
                objGpet.sinogram_size.span = -1;
                objGpet.sinogram_size.nRings = 1;
                objGpet.sinogram_size.maxRingDifference = 0;
            end
            if isempty(objGpet.nSubsets)
                objGpet.nSubsets = 1;
            end
            if isempty(objGpet.nIter)
                objGpet.nIter = 40;
            end
            objGpet.sinogram_size.matrixSize = [objGpet.sinogram_size.nRadialBins objGpet.sinogram_size.nAnglesBins objGpet.sinogram_size.nSinogramPlanes];
            objGpet.osem_subsets(objGpet.nSubsets, objGpet.sinogram_size.nAnglesBins);
        end
                        
        function G_mMR_setup(objGpet)
            % Default parameter, only if it havent been loaded by the
            % config previously:
            if isempty(objGpet.sinogram_size)
                objGpet.sinogram_size.nRadialBins = 344;
                objGpet.sinogram_size.nAnglesBins = 252;
                objGpet.sinogram_size.nRings = 64;
                objGpet.sinogram_size.nSinogramPlanes = 837;
                objGpet.sinogram_size.maxRingDifference = 60;
                objGpet.sinogram_size.nPlanesPerSeg = [127   115   115    93    93    71    71    49    49    27    27];
                objGpet.sinogram_size.span = 11;
                objGpet.sinogram_size.nSeg = 11;
                objGpet.scanner_properties.radius_mm = 328;
                objGpet.scanner_properties.sinogramDepthOfInteraction_mm = 6.7;
                objGpet.scanner_properties.LORDepthOfInteraction_mm = 9.6;
                objGpet.scanner_properties.planeSep_mm = 2.03125;
                objGpet.scanner_properties.nCrystalsPerRing = 504;                
                objGpet.scanner_properties.binSize_mm = pi*objGpet.scanner_properties.radius_mm/objGpet.scanner_properties.nCrystalsPerRing;%2.0445;

            else
                
            end
            if isempty(objGpet.nSubsets)
                objGpet.nSubsets = 21;
            end
            if isempty(objGpet.nIter)
                objGpet.nIter = 3;
            end
            
            objGpet.sinogram_size.matrixSize = [objGpet.sinogram_size.nRadialBins objGpet.sinogram_size.nAnglesBins objGpet.sinogram_size.nSinogramPlanes];
            objGpet.image_size.matrixSize =[344, 344, 127];
            objGpet.image_size.voxelSize_mm = [2.08626 2.08626 2.03125];
            
            objGpet.osem_subsets(objGpet.nSubsets, objGpet.sinogram_size.nAnglesBins);
        end
        
        function G_cylindrical_setup(objGpet)
            % Default parameter, only if it havent been loaded by the
            % config previously. By default use the same size of mmr, additionally needs the scanner radius:
            
            if isempty(objGpet.sinogram_size)
                objGpet.scanner_properties.radius_mm = 356;
                objGpet.sinogram_size.nRadialBins = 344;
                objGpet.sinogram_size.nAnglesBins = 252;
                objGpet.sinogram_size.nRings = 64;
                objGpet.sinogram_size.nSinogramPlanes = 837;
                objGpet.sinogram_size.maxRingDifference = 60;
                objGpet.sinogram_size.nPlanesPerSeg = [127   115   115    93    93    71    71    49    49    27    27];
                objGpet.sinogram_size.span = 11;
                objGpet.sinogram_size.nSeg = 11;
                %add maxDiff
                %                 objGpet.radialBinTrim = 0;
                           
            end
            
            if ~isfield(objGpet.scanner_properties, 'radius_mm')
                objGpet.scanner_properties.radius_mm = 370;
            end
            
            if isempty(objGpet.nSubsets)
                objGpet.nSubsets = 21;
            end
            if isempty(objGpet.nIter)
                objGpet.nIter = 3;
            end
            
            objGpet.sinogram_size.matrixSize = [objGpet.sinogram_size.nRadialBins objGpet.sinogram_size.nAnglesBins objGpet.sinogram_size.nSinogramPlanes];
            objGpet.image_size.matrixSize =[344, 344, 127];
            objGpet.image_size.voxelSize_mm = [2.08626 2.08626 2.03125];
            
            objGpet.osem_subsets(objGpet.nSubsets, objGpet.sinogram_size.nAnglesBins);
            
            % Init the scanner parameters depending on the desired image.
            % The FOV used covers the whole image:
            objGpet.scanner_properties.radialFov_mm = max(objGpet.image_size.matrixSize(2)*objGpet.image_size.voxelSize_mm(2),objGpet.image_size.matrixSize(1)*objGpet.image_size.voxelSize_mm(1))./2;
            objGpet.scanner_properties.axialFov_mm = objGpet.image_size.matrixSize(3)*objGpet.image_size.voxelSize_mm(3);
        end
        
        function set_subsets(objGpet, numSubsets)
            % Update number of iteration to keep the subset%iterations
            % constant:
            objGpet.nIter = objGpet.nIter*objGpet.nSubsets/numSubsets;
            objGpet.nSubsets = numSubsets;
            
            objGpet.osem_subsets(objGpet.nSubsets, objGpet.sinogram_size.nAnglesBins);
        end
        
        function objGpet=init_image_properties(objGpet, refImage)
            objGpet.image_size.matrixSize = refImage.ImageSize;
            
            if refImage.ImageSize(3) >1
                objGpet.image_size.voxelSize_mm = [refImage.PixelExtentInWorldY refImage.PixelExtentInWorldX refImage.PixelExtentInWorldZ];
            else
                objGpet.image_size.voxelSize_mm = [refImage.PixelExtentInWorldY refImage.PixelExtentInWorldX];
            end
        end
        
        function objGpet =init_ref_image(objGpet)
            % Image centred:
            origin_mm = [-objGpet.image_size.voxelSize_mm(2)*objGpet.image_size.matrixSize(2)/2 -objGpet.image_size.voxelSize_mm(1)*objGpet.image_size.matrixSize(1)/2 ...
                -objGpet.image_size.voxelSize_mm(3)*objGpet.image_size.matrixSize(3)/2];
            XWorldLimits= [origin_mm(2) origin_mm(2)+objGpet.image_size.voxelSize_mm(2)*objGpet.image_size.matrixSize(2)];
            YWorldLimits= [origin_mm(1) origin_mm(1)+objGpet.image_size.voxelSize_mm(1)*objGpet.image_size.matrixSize(1)];
            ZWorldLimits= [-objGpet.image_size.voxelSize_mm(3)*objGpet.image_size.matrixSize(3)/2 objGpet.image_size.voxelSize_mm(3)*objGpet.image_size.matrixSize(3)/2] + [objGpet.bed_position_mm objGpet.bed_position_mm];
            refImage = imref3d(objGpet.image_size.matrixSize, XWorldLimits, YWorldLimits, ZWorldLimits);
            objGpet.ref_image = refImage;
        end
        
        % This functions get the field of the struct used in Apirl and
        % converts to the struct sinogram_size used in this framework.
        function structSizeSino = get_sinogram_size_for_apirl(objPETRawData)
            structSizeSino.numR = objPETRawData.sinogram_size.nRadialBins;
            structSizeSino.numTheta = objPETRawData.sinogram_size.nAnglesBins;
            structSizeSino.numSinogramPlanes = objPETRawData.sinogram_size.nSinogramPlanes;
            structSizeSino.span = objPETRawData.sinogram_size.span;
            structSizeSino.numZ = objPETRawData.sinogram_size.nRings;
            structSizeSino.rFov_mm = 0; % not used
            structSizeSino.zFov_mm = 0; % not used
            if structSizeSino.span > 0
                structSizeSino.numSegments = objPETRawData.sinogram_size.nSeg;
                structSizeSino.sinogramsPerSegment = objPETRawData.sinogram_size.nPlanesPerSeg;
                structSizeSino.minRingDiff = objPETRawData.sinogram_size.minRingDiffs;
                structSizeSino.maxRingDiff = objPETRawData.sinogram_size.maxRingDiffs;
                structSizeSino.maxAbsRingDiff = objPETRawData.sinogram_size.maxRingDifference;
                structSizeSino.numPlanesMashed = objPETRawData.sinogram_size.numPlanesMashed;
            end
        end
        
    end
    
    % Methods in a separate file:
    methods (Access = private)
        
        ii = bit_reverse(objGpet, mm);
        lambda = Project_preComp(objGpet,X,g,Angles,RadialBins,dir);
        g = init_precomputed_G (objGpet);
        
        % SAM ELLIS EDIT (18/07/2016): new method to allow easy division without
        % using a small additive term to avoid div. by zero. 
        function c = vecDivision(objGpet,a,b)
            % element-by-element division of two vectors, a./b, BUT avoiding
            % division by 0
            c = a;
            c(b~=0) = a(b~=0)./b(b~=0);
            c(b==0) = 0;
        end
        
        function objGpet =  varargin_pair(objGpet, varargs)
            % example PET = classGpet('scanner','mMR',
            % 'nSubsets',14,'PSF.Width', 2);
            
            npair = floor(length(varargs) / 2);
            if 2*npair ~= length(varargs),
                error('need names and values in pairs');
            end
            args = {varargs{1:2:end}};
            vals = {varargs{2:2:end}};
            
            prop = properties(objGpet);
            
            for ii=1:npair
                arg = args{ii};
                val = vals{ii};
                
                if ~ischar(arg)
                    error('unknown option of class %s', class(arg))
                end
                
                if sum(strcmpi(prop, arg )) > 0
                    objGpet.(arg) = val;
                    continue;
                end
                
                % for pairs with sub-fields such as : ('PSF.Width', 3)
                idot = strfind(arg, '.');
                if ~isempty(idot)
                    arg1 = arg([1:(idot-1)]);
                    arg2 = arg([(idot+1):end]);
                    
                    if ~(sum(strcmpi(prop, arg1 )) > 0)
                       error('unknown property')
                    end
                    subProp = fieldnames(objGpet.(arg1));
                    if ~(sum(strcmpi(subProp, arg2 )) > 0)
                        error('unknown sub-property')
                    end
                    s = struct('type', {'.', '.'}, 'subs', ...
                        { arg([1:(idot-1)]), arg([(idot+1):end]) });
                    
                    objGpet = subsasgn(objGpet, s, val);
                end
                
            end
            
        end
        
        function nPlanePerSeg = Planes_Seg (objGpet)
            
            a = (objGpet.sinogram_size.maxRingDifference -(objGpet.sinogram_size.span+1)/2);
            b = floor(a/objGpet.sinogram_size.span)+ floor(objGpet.sinogram_size.maxRingDifference/a);
            if b > 0
                nseg = 2*b+1;
            else
                nseg = 1;
            end
            
            a = ones(objGpet.sinogram_size.nRings);
            minRingDiff = [0,(objGpet.sinogram_size.span+1)/2:objGpet.sinogram_size.span:objGpet.sinogram_size.maxRingDifference];

            s = zeros(1,(nseg+1)/2);
            for j = 1:(nseg+1)/2
                s(j) = length(diag(a,minRingDiff(j)));
            end

            if objGpet.sinogram_size.span>1, s = 2*s-1; end

            nPlanePerSeg = zeros(1,nseg);
            nPlanePerSeg(1) = s(1);
            nPlanePerSeg(2:2:end) = s(2:end);
            nPlanePerSeg(3:2:end) = s(2:end);

        end

        function Scatter3D = iSSRB(objGpet,Scatter2D)
            
            nPlanePerSeg = Planes_Seg(objGpet);
            
            mo = cumsum(nPlanePerSeg)';
            no =[[1;mo(1:end-1)+1],mo];            

            Scatter3D = zeros(objGpet.sinogram_size.matrixSize,'single');
            % For span 1 this method fails, we use the other available in
            % the library for any span conversion:
            if objGpet.sinogram_size.span > 1

                Scatter3D(:,:,no(1,1):no(1,2),:) = Scatter2D;
            
                for i = 2:2:length(nPlanePerSeg)

                    delta = (nPlanePerSeg(1)- nPlanePerSeg(i))/2;
                    indx = nPlanePerSeg(1) - delta;

                    Scatter3D (:,:,no(i,1):no(i,2),:) = Scatter2D(:,:,delta+1:indx,:);
                    Scatter3D (:,:,no(i+1,1):no(i+1,2),:) = Scatter2D(:,:,delta+1:indx,:);
                end
            else
                % issrb is for an input sinogram span of span 121
                structSizeSino3d = getSizeSino3dFromSpan(objGpet.sinogram_size.nRadialBins, objGpet.sinogram_size.nAnglesBins, objGpet.sinogram_size.nRings, ...
                    0, 0, 121, objGpet.sinogram_size.maxRingDifference);
                [Scatter3D, structSizeSino3dSpanN] = convertSinogramToSpan(Scatter2D, structSizeSino3d,  objGpet.sinogram_size.span);
            end
        end
        
        function scatter_3D = scatter_scaling(objGpet,scatter_3D, emission_sinogram, ncf, acf, randoms)
            
            gaps = ncf==0;
            scatter_3D =  scatter_3D./ncf;
            scatter_3D(gaps)=0;
            
            Trues = emission_sinogram - randoms;     
            Trues(Trues<0) = 0;
            for i = 1: size(acf,3)
                acf_i = acf(:,:,i);
                %                 mask = acf_i <= min(acf(:));
                mask = acf_i <1.03;
                scatter_3D_tail = scatter_3D(:,:,i).*mask;
                Trues_tail = Trues(:,:,i).*mask;
                
                scale_factor = sum(Trues_tail(:))./sum(scatter_3D_tail(:));
                scatter_3D(:,:,i) = scatter_3D(:,:,i)/scale_factor;
                
            end
            
        end
        
    end
    methods (Access = public)
        % Project:
        m = P(objGpet, x,subset_i, localNumSubsets);
        % Backproject:
        x = PT(objGpet,m, subset_i, localNumSubsets);
        % Normalization correction factors:
        [n, n_ti, n_tv, gaps] = NCF(varargin);
        % Attenuation correction factors:
        a=ACF(varargin);
        % Randoms:
        r=R(varargin);
        % Scatter
        s=S(varargin);
        %
        osem_subsets(objGpet, nsub,nAngles);
        
        function gaps = gaps(objGpet)
            if strcmp(objGpet.scanner,'mMR')
                if objGpet.sinogram_size.span > 0
                    crystalMasks = ones(504,64);
                    crystalMasks(9:9:end,:) = 0;
                    gaps = createSinogram3dFromDetectorsEfficency(crystalMasks, objGpet.get_sinogram_size_for_apirl(), 0);
                else
                    crystalMasks = ones(504);
                    crystalMasks(9:9:end,:) = 0;
                    gaps = createSinogram2dFromDetectorsEfficency(crystalMasks, objGpet.get_sinogram_size_for_apirl(), 1, 0);
                end
            else
                gaps = ones(objGpet.sinogram_size.matrixSize);
            end
        end
        function x = ones(objGpet)
            if objGpet.sinogram_size.span == -1
                %x = ones([objGpet.image_size.matrixSize(1) objGpet.image_size.matrixSize(2)]);
                [x0,y0] = meshgrid(-objGpet.image_size.matrixSize(1)/2+1:objGpet.image_size.matrixSize(2)/2);
                x = (x0.^2+y0.^2)<(objGpet.image_size.matrixSize(1)/2.5)^2;
            else
                [x0,y0] = meshgrid(-objGpet.image_size.matrixSize(1)/2+1:objGpet.image_size.matrixSize(2)/2);
                x = (x0.^2+y0.^2)<(objGpet.image_size.matrixSize(1)/2.5)^2;
                x = repmat(x,[1,1,objGpet.image_size.matrixSize(3)]);
            end
            x = single(x);
        end
        
        function x = zeros(objGpet)
            
            x = zeros(objGpet.image_size.matrixSize);
        end
        
        init_sinogram_size(objGpet, inSpan, numRings, maxRingDifference);
        
        function sino_size = get_sinogram_size(objGpet)
            sino_size = objGpet.sinogram_size;
        end
        
        function setBedPosition(objGpet, siemensInterfile)
            [auxImage, auxRef, objGpet.bed_position_mm] = interfileReadSiemensImage(siemensInterfile);
            objGpet.init_ref_image();
        end
        
        % Function that maps MR into PET image space:
        function [MrInPet, refResampledImage] = getMrInPetImageSpace(objGpet, pathMrDicom)
            % Read dicom image:
            [imageMr, refMrImage, affineMatrix, dicomInfo] = ReadDicomImage(pathMrDicom, '', 1);
            % Convert into the nes image space
            [MrInPet, refResampledImage] = ImageResample(imageMr, refMrImage, objGpet.ref_image);
        end
        
        % Function that reads MR and  configures PET to reconstruct in that space:
        function [imageMr, refImageMr, MrInPetFov, refImagePet] = getMrInNativeImageSpace(objGpet, pathMrDicom)
            % Read dicom image:
            [imageMr, refImageMr, affineMatrix, dicomInfo] = ReadDicomImage(pathMrDicom, '', 1);
            % Update the ref_image to this pixel size:
            newVoxelSize= [refImageMr.PixelExtentInWorldY refImageMr.PixelExtentInWorldX refImageMr.PixelExtentInWorldZ];
            ratio = objGpet.image_size.voxelSize_mm ./ newVoxelSize;
            objGpet.image_size.voxelSize_mm = newVoxelSize;
            objGpet.image_size.matrixSize = round(objGpet.image_size.matrixSize .* ratio);
            objGpet.init_ref_image();
            refImagePet = objGpet.ref_image;
            % And finally complete the MR image into PET FOV:
            [MrInPetFov, refImageMrFov] = ImageResample(imageMr, refImageMr, objGpet.ref_image);    
        end
        
        % Converts sinogram
        function [sino_compressed, structSizeSino3dSpanN] = apply_axial_compression_from_span1(objGpet, sinogram)
            structSizeSino3d = getSizeSino3dFromSpan(objGpet.sinogram_size.nRadialBins, objGpet.sinogram_size.nAnglesBins, objGpet.sinogram_size.nRings, ...
                0, 0, 1, objGpet.sinogram_size.maxRingDifference);
            if numel(sinogram) ~= (objGpet.sinogram_size.nRadialBins*objGpet.sinogram_size.nAnglesBins*sum(structSizeSino3d.sinogramsPerSegment))
                error('The input sinogram needs to be span 1.');
            end
            [sino_compressed, structSizeSino3dSpanN] = convertSinogramToSpan(sinogram, structSizeSino3d,  objGpet.sinogram_size.span);
        end
        
        function mask = get_fov_maks(objGpet, radiusFov_mm)
            if nargin == 1
                radiusFov_mm = 596/2;
            end
            mask = objGpet.ones;
            radiusFov_pixels = radiusFov_mm/objGpet.image_size.voxelSize_mm(1);
            [X,Y,Z] = meshgrid(1:objGpet.image_size.matrixSize(2),1:objGpet.image_size.matrixSize(1),1:objGpet.image_size.matrixSize(3));
            indicesOutMask = ((X-objGpet.image_size.matrixSize(2)/2).^2+(Y-objGpet.image_size.matrixSize(1)/2).^2) > radiusFov_pixels.^2;
            mask(indicesOutMask) = 0;
        end
        
        function SenseImg = Sensitivity(objGpet, AN)
            % SAM ELLIS EDIT: IF AN IS DOUBLE, THEN LET THE SENSEIMG BE
            % DOUBLE TOO
            classAN = whos('AN');
            classAN = classAN.class;
            
            SenseImg = zeros([objGpet.image_size.matrixSize, objGpet.nSubsets],classAN) ;
            % SAM ELLIS EDIT (23/08/2016): only show messages if using more
            % than one subset
            if objGpet.nSubsets > 1
                for n = 1:objGpet.nSubsets
                    fprintf('%d, ',n);
                    SenseImg(:,:,:,n) = objGpet.PT(AN,n);
                end
            else
                SenseImg = objGpet.PT(AN);
            end
            fprintf('Done.\n');
        end
        
        function display(objGpet) %#ok<DISPLAY>
            disp(objGpet)
            methods(objGpet)
        end
        
        function objGpet = Revise(objGpet,opt)
            % to revise the properties of a given object without
            % re-instantiation
            vfields = fieldnames(opt);
            prop = properties(objGpet);
            for i = 1:length(vfields)
                % if is an struct, update only the received fields in that
                % structure, for example for sinogram_size:
                if isstruct(opt.(vfields{i}))
                    field = vfields{i};
                    if sum(strcmpi(prop, field )) > 0
                        subprop = fieldnames(objGpet.(field));
                        secondary_fields = fieldnames(opt.(field));
                        for j = 1 : length(secondary_fields)
                            subfield = secondary_fields{j};
                            if sum(strcmpi(subprop, subfield )) > 0
                                objGpet.(field).(secondary_fields{j}) = opt.(field).(secondary_fields{j});
                            end
                        end
                    end
                else
                    % It isnt
                    field = vfields{i};
                    if sum(strcmpi(prop, field )) > 0
                        objGpet.(field) = opt.(field);
                    end
                end
            end
            objGpet.init_sinogram_size(objGpet.sinogram_size.span, objGpet.sinogram_size.nRings, objGpet.sinogram_size.maxRingDifference);
            if isfield(opt,'nSubsets')
                objGpet.set_subsets(opt.nSubsets)
            end
            
            % SAM ELLIS EDIT (23/08/2016): for 2D radon, need to update
            % sinogram size based on new image size
            if any(strcmpi(vfields,'image_size')) && strcmpi(objGpet.scanner,'2D_radon')
                x = radon(ones(objGpet.image_size.matrixSize(1:2)),0:179);
                x = size(x);
                objGpet.sinogram_size.nRadialBins = x(1);
                objGpet.sinogram_size.nAnglesBins = x(2);
                objGpet.sinogram_size.nSinogramPlanes = 1;
                objGpet.sinogram_size.span = -1;
                objGpet.sinogram_size.nRings = 1;
                objGpet.sinogram_size.maxRingDifference = 0;
            end
        end
        
        function Img = OPMLEM(objGpet,Prompts,RS, SensImg,Img, nIter)
            for i = 1:nIter
		% SAM ELLIS EDIT (18/07/2016): replaced vector divisions by vecDivision
                Img = Img.*objGpet.vecDivision(objGpet.PT(objGpet.vecDivision(Prompts,objGpet.P(Img)+ RS)),SensImg);
                Img = max(0,Img);
            end
        end
        
        function image = OPMLEMsaveIter(objGpet,Prompts, AN, RS, SensImg, initialEstimate, nIter, outputPath, saveInterval)
            if ~isdir(outputPath)
                mkdir(outputPath);
            end
            image{1} = initialEstimate;
            for i = 1:nIter
		% SAM ELLIS EDIT (18/07/2016): replaced vector divisions by vecDivision
                image{i+1} = image{i}.*objGpet.vecDivision(objGpet.PT(AN.*objGpet.vecDivision(Prompts,AN.*objGpet.P(image{i})+ RS)),SensImg);
                image{i+1} = max(0,image{i+1});
                if rem(i-1,saveInterval) == 0 % -1 to save the first iteration
                    interfilewrite(single(image{i+1}), [outputPath 'mlem_iter_' num2str(i)], objGpet.image_size.voxelSize_mm); % i use i instead of i+1 because i=1 is the inital estimate
                end
            end
        end
        
        function image = OPOSEMsaveIter(objGpet,Prompts, AN, RS, SensImg, initialEstimate, nIter, outputPath, saveInterval)
            k=1;
            image{k} = initialEstimate;
            for i = 1:nIter
                for j = 1:objGpet.nSubsets
                    % SAM ELLIS EDIT (18/07/2016): replaced vector divisions by vecDivision
                    image{k+1} = image{k}.*objGpet.vecDivision(objGpet.PT(AN.*objGpet.vecDivision(Prompts,AN.*objGpet.P(image{k},j)+ RS),j),SensImg(:,:,:,j));
                    image{k+1} = max(0,image{k+1});
                    if rem(k-1,saveInterval) == 0 % -1 to save the first iteration
                        interfilewrite(single(image{k+1}), [outputPath 'opmlem_iter_' num2str(k)], objGpet.image_size.voxelSize_mm); % i use i instead of i+1 because i=1 is the inital estimate
                    end
                    k = k + 1;
                end
            end
        end
        
        function Img = OPOSEM(objGpet,Prompts,RS, SensImg,Img, nIter)
            for i = 1:nIter
                for j = 1:objGpet.nSubsets
                    % SAM ELLIS EDIT (18/07/2016): replaced vector divisions by vecDivision
                    Img = Img.*objGpet.vecDivision(objGpet.PT(objGpet.vecDivision(Prompts,objGpet.P(Img,j)+ RS),j),SensImg(:,:,:,j));
                    % Img = Img.*objGpet.PT(Prompts./(objGpet.P(Img,j)+ RS + 1e-5),j)./(SensImg(:,:,:,j)+1e-5);
                    Img = max(0,Img);
                end
            end
        end
        
        % SAM ELLIS EDIT: new sensitivity image for basis function
        % coefficients
        function alphSens = basisSensitivity(objGpet,AN,basisMat)
            
            % define function to multiply image by matrix and reshape
            % (assuming numel(alph) ==  numel(Img))
            multiplyByBases = @(alphIn,basisMat) reshape(basisMat*double(alphIn(:)),size(alphIn));
            
            % get sensitivity image, using input att.*norm factors
            SensImg = Sensitivity(objGpet, AN);
            
            % calcuate new sens image in terms of the alpha coefficients
            alphSens = multiplyByBases(SensImg,basisMat');
        end
        
        % SAM ELLIS EDIT: new iterative method using a basis function matrix (full matrix required)
        function [alph,Img] = BFEM(objGpet,basisMat,Prompts,RS, AN, alphSens, alph, nIter)
            % assume the basisFcns are in a matrix for now..
            
            % define function to multiply image by matrix and reshape
            % (assuming numel(alph) ==  numel(Img))
            multiplyByBases = @(alphIn,basisMat) reshape(basisMat*double(alphIn(:)),size(alphIn));
     
            for ii = 1:nIter
                
                % forward model:
                FP_guess = AN.*objGpet.P(multiplyByBases(alph,basisMat)) + RS;
                
               % ratio with measured data and backproject (including
               % transpose basis function multiplication)
               corr_fact = multiplyByBases(objGpet.PT(AN.*objGpet.vecDivision(Prompts,FP_guess)),basisMat');
               
               % divide by sens image and multiply by old alpha values
               alph = alph.*objGpet.vecDivision(corr_fact,alphSens);
            end
            
            Img = multiplyByBases(alph,basisMat);
            
        end
        
        % SAM ELLIS EDIT (24/08/2016): new method to delete the temp folder
        % when we're done
        function delete(objGpet)
            if (objGpet.deleteTemp ~= false) && (objGpet.deleteTemp ~= 0)
                % if the flag is not false, then it is true, and delete the
                % temp folder and its contents if it exists
                if exist(objGpet.tempPath,'dir') == 7 
                    rmdir(objGpet.tempPath,'s');
                end
            else
            end
        end
        
        gf3d = Gauss3DFilter (objGpet, data, fwhm);
        [Img,totalScaleFactor, info] = BQML(objGpet,Img,sinogramInterFileFilename,normalizationInterFileFilename);
        Img = SUV(objGpet,Img,sinogramInterFileFilename,normalizationInterFileFilename);
    end
    
end


