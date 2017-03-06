classdef classGpet < handle
    properties (SetAccess = private)
        % Type of scanner. Options: '2D_radon', '2D_mMR', 'mMR'
        scanner 
        % Sinogram size:
        sinogram_size     % Struct with the size of the sinogram
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
    end
    
    methods
        % Constructors:
        function objGpet = classGpet(varargin) % Default options: (). From file (filename)

            objGpet.scanner = 'mMR';
            objGpet.method =  'otf_siddon_cpu';
            objGpet.PSF.type = 'none';
            objGpet.radialBinTrim = 0;
            objGpet.Geom = '';
            objGpet.tempPath = './temp/'; 
              
           if nargin == 1
                % Read configuration from file or from struct:
                if isstruct(varargin{1})
                    if ~isfield(varargin{1}, 'scanner') || ~isfield(varargin{1}, 'method')
                        disp('Configuration for ''mMR'' scanner and ''otf_siddon_cpu''');
                    end
                    % get fields from user's param
                        vfields = fieldnames(varargin{1});
                        prop = properties(objGpet);
                        for i = 1:length(vfields)
                            field = vfields{i};
                            if sum(strcmpi(prop, field )) > 0
                                objGpet.(field) = varargin{1}.(field);
                            end
                        end
                elseif ischar(varargin{1})
                    objGpet.readConfigFromFile(varargin{1})
                end
            else
                % Other options?
            end
            if ~isdir(objGpet.tempPath)
                mkdir(objGpet.tempPath)
            end
            % Init scanner properties:
            objGpet.initScanner();
        end
        
        function objGpet = readConfigFromFile(objGpet, strFilename)
            error('todo: read configuration from file');
        end
        
        % Function that intializes the scanner:
        function initScanner(objGpet)
            if strcmpi(objGpet.scanner,'2D_radon')
                objGpet.G_2D_radon_setup();
            elseif strcmpi(objGpet.scanner,'mMR')
                objGpet.G_mMR_setup();
            elseif strcmpi(objGpet.scanner,'2D_mMR')
                objGpet.G_2D_mMR_setup();
            else
                error('unkown scanner')
            end
        end
        
        function x = ones(objGpet)
            x = ones(objGpet.image_size.matrixSize);
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
        
        function G_2D_mMR_setup(objGpet)
            if isempty(objGpet.sinogram_size)
                objGpet.sinogram_size.nRadialBins = 344;
                objGpet.sinogram_size.nAnglesBins = 252;
                objGpet.sinogram_size.nSinogramPlanes = 1;
                objGpet.sinogram_size.span = 1;
                objGpet.sinogram_size.nSeg = 1;
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
            objGpet.image_size.matrixSize =[344, 344, 1];
            objGpet.image_size.voxelSize_mm = [2.08626 2.08626 2.03125];
            
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
                objGpet.sinogram_size.nPlanesPerSeg = [127   115   115    93    93    71    71    49    49    27    27];
                objGpet.sinogram_size.span = 11;
                objGpet.sinogram_size.nSeg = 11;
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
        
        function set_subsets(objGpet, numSubsets)
            % Update number of iteration to keep the subset%iterations
            % constant:
            objGpet.nIter = objGpet.nIter*objGpet.nSubsets/numSubsets;
            objGpet.nSubsets = numSubsets;
            
            objGpet.osem_subsets(objGpet.nSubsets, objGpet.sinogram_size.nAnglesBins);
        end
        
        function objGpet=init_image_properties(objGpet, refImage)
            objGpet.image_size.matrixSize = refImage.ImageSize;
            objGpet.image_size.voxelSize_mm = [refImage.PixelExtentInWorldY refImage.PixelExtentInWorldX refImage.PixelExtentInWorldZ];
        end
        
    end
    
    % Methods in a separate file:
    methods (Access = private)
        gf3d = Gauss3DFilter (objGpet, data, image_size, fwhm);
        ii = bit_reverse(objGpet, mm);
        lambda = Project_preComp(objGpet,X,g,Angles,RadialBins,dir);
        init_sinogram_size(objGpet, inSpan, numRings, maxRingDifference);
        g = init_precomputed_G (objGpet);
    end
    methods (Access = public)
        m = P(objGpet, x,subset_i);
        x = PT(objGpet,m, subset_i);
        n=NCF(varargin);
        osem_subsets(objGpet, nsub,nAngles);
    end
end
