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
        % Asymmertic pre-computed sysetm matrix
        Geom
        % Method to estimate randoms:
        method_for_randoms
        % Method to estimate scatter:
        method_for_scatter
        % Operative system.
        os
        % bar for the paths.
        bar
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
                elseif any(cellfun(@any,strfind(varargin,'scanner')))
                    i = find(cellfun(@any,strfind(varargin,'scanner')));
                    objGpet.scanner = varargin{i+1};
                else
                    objGpet.scanner = 'mMR';
                end
            end
            objGpet.initScanner();
            objGpet.method =  'otf_siddon_cpu';
            objGpet.PSF.type = 'shift-invar';
            objGpet.PSF.Width = 4; %mm
            objGpet.radialBinTrim = 0;
            objGpet.Geom = '';
            objGpet.tempPath = ['.' objGpet.bar 'temp' objGpet.bar]; 
            objGpet.method_for_randoms = 'from_ML_singles_matlab';  
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
                %add maxDiff
                %                 objGpet.radialBinTrim = 0;
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
        
        % This functions get the field of the struct used in Apirl and
        % converts to the struct sinogram_size used in this framework.
        function structSizeSino = get_sinogram_size_for_apirl(objPETRawData)
            structSizeSino.numR = objPETRawData.sinogram_size.nRadialBins;
            structSizeSino.numTheta = objPETRawData.sinogram_size.nAnglesBins;
            structSizeSino.numSinogramPlanes = objPETRawData.sinogram_size.nSinogramPlanes;
            structSizeSino.span = objPETRawData.sinogram_size.span;
            structSizeSino.numZ = objPETRawData.sinogram_size.nRings;
            if structSizeSino.span > 0
                structSizeSino.numSegments = objPETRawData.sinogram_size.nSeg;
                structSizeSino.sinogramsPerSegment = objPETRawData.sinogram_size.nPlanesPerSeg;
                structSizeSino.minRingDiff = objPETRawData.sinogram_size.minRingDiffs;
                structSizeSino.maxRingDiff = objPETRawData.sinogram_size.maxRingDiffs;
                structSizeSino.numPlanesMashed = objPETRawData.sinogram_size.numPlanesMashed;
            end
        end
        
    end
    
    % Methods in a separate file:
    methods (Access = private)
        gf3d = Gauss3DFilter (objGpet, data, image_size, fwhm);
        ii = bit_reverse(objGpet, mm);
        lambda = Project_preComp(objGpet,X,g,Angles,RadialBins,dir);
        g = init_precomputed_G (objGpet);
        
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
            nseg = 2*b+1;

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
            Scatter3D(:,:,no(1,1):no(1,2),:) = Scatter2D;
            
            for i = 2:2:length(nPlanePerSeg)
                
                delta = (nPlanePerSeg(1)- nPlanePerSeg(i))/2;
                indx = nPlanePerSeg(1) - delta;
                
                Scatter3D (:,:,no(i,1):no(i,2),:) = Scatter2D(:,:,delta+1:indx,:);
                Scatter3D (:,:,no(i+1,1):no(i+1,2),:) = Scatter2D(:,:,delta+1:indx,:);
            end
        end
        
        function scatter_3D = scatter_scaling(objGpet,scatter_3D, emission_sinogram, ncf, acf, randoms)
            
            gaps = ncf==0;
            scatter_3D =  scatter_3D./ncf;
            scatter_3D(gaps)=0;
            
            Trues = emission_sinogram - randoms;     
            
            for i = 1: size(acf,3)
                acf_i = acf(:,:,i);
                mask = acf_i <= min(acf(:));
                
                scatter_3D_tail = scatter_3D(:,:,i).*mask;
                Trues_tail = Trues(:,:,i).*mask;
                
                scale_factor = sum(Trues_tail(:))./sum(scatter_3D_tail(:));
                scatter_3D(:,:,i) = scatter_3D(:,:,i)*scale_factor;
            end
            
        end
                       
    end
    methods (Access = public)
        % Project:
        m = P(objGpet, x,subset_i);
        % Backproject:
        x = PT(objGpet,m, subset_i);
        % Normalization correction factors:
        [n, n_ti, n_tv] = NCF(varargin);
        % Attenuation correction factors:
        a=ACF(varargin);
        % Randoms:
        r=R(varargin);
        % Scatter
        s=S(varargin);
        %
        osem_subsets(objGpet, nsub,nAngles);
        
        function x = ones(objGpet)
            [x0,y0] = meshgrid(-objGpet.image_size.matrixSize(1)/2+1:objGpet.image_size.matrixSize(2)/2);
            x = (x0.^2+y0.^2)<(objGpet.image_size.matrixSize(1)/2.5)^2;
            x = repmat(x,[1,1,objGpet.image_size.matrixSize(3)]);
        end
        
        init_sinogram_size(objGpet, inSpan, numRings, maxRingDifference);
        
        function sino_size = get_sinogram_size(objGpet)
            sino_size = objGpet.sinogram_size;
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
        
        
        function SenseImg = Sensitivity(objGpet, AN)
            SenseImg = zeros([objGpet.image_size.matrixSize, objGpet.nSubsets],'single') ;
            fprintf('Subset: ');
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
        end
        
        function Img = OPMLEM(objGpet,Prompts,RS, SensImg,Img, nIter)
            for i = 1:nIter
                Img = Img.*objGpet.PT(Prompts./(objGpet.P(Img)+ RS + 1e-5))./(SensImg+1e-5);
                Img = max(0,Img);
            end
        end
        
        function Img = OPOSEM(objGpet,Prompts,RS, SensImg,Img, nIter)
            for i = 1:nIter
                for j = 1:objGpet.nSubsets
                    Img = Img.*objGpet.PT(Prompts./(objGpet.P(Img,j)+ RS + 1e-5),j)./(SensImg(:,:,:,j)+1e-5);
                    Img = max(0,Img);
                end
            end
        end
        

    end
    
end


