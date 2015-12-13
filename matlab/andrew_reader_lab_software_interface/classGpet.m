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
      % Temporary files path:
      tempPath
    end
   
    methods
        % Constructors:
        function objGpet = classGpet(varargin) % Default options: (). From file (filename)
            if nargin == 0
                objGpet.scanner = 'mMR'; 
                objGpet.method =  'otf_siddon_cpu';
                objGpet.PSF.type = 'none';
                objGpet.tempPath = './temp/';
                if ~isdir(objGpet.tempPath)
                    mkdir(objGpet.tempPath)
                end
            elseif nargin == 1
                % Read configuration from file or from struct:
                if isstruct(varargin{1})
                    if ~isfield(varargin{1}, 'scanner') || ~isfield(varargin{1}, 'method')
                        error('The parameters scanner and method are mandatory for the input configuration structure');
                    end
                    % get fields from user's param
                    param.null = 0;
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
        
        % Forward projector:
        function m = P(objGpet, x,subset_i)
            % full/sub-forward
            if nargin <3
                angles = 0:objGpet.sinogram_size.nAnglesBins-1;
            else
                angles = objGpet.sinogram_size.subsets(:,subset_i);
            end

            % PSF convolution
            if strcmpi(objGpet.PSF.type,'shift-invar')
                    x = Gauss3DFilter(x, objGpet.image_size, objGpet.PSF.Width);
            else
                disp('todo: shift-var')
            end

            if strcmpi(objGpet.scanner,'2D_radon')
                if strcmpi(objGpet.method, 'otf_matlab')
                    m = radon(x,angles);
                else
                    error(sprintf('The method %s is not available for the scanner %s.', objGpet.method, objGpet.scanner));
                end
            elseif strcmpi(objGpet.scanner,'mMR')||strcmpi(objGpet.scanner,'2D_mMR')
                % Check the image size:
                if size(x) == 1
                    % Uniform image with x value:
                    x = ones(objGpet.image_size.matrixSize).*x;
                else
                    sizeX = size(x);
                    if numel(sizeX) == 2
                        sizeX = [size(x) 1];% 2d, I need to add the z size because
                        % matrixSize is a 3-elements vector.
                    end
                    if sizeX ~= objGpet.image_size.matrixSize
                        warning('x: the input image has a different size to the matrix_size of the proejctor');
                    end
                end
                if strcmpi(objGpet.scanner,'mMR')
                    if strcmpi(objGpet.method, 'pre-computed_matlab') 
                        % G = mMR_forward(x,G,angles); using G.method
                    else
                        % Select the subsets:
                        if nargin < 3
                            subset_i = [];
                            numSubsets = []; 
                        else
                            numSubsets = objGpet.nSubsets;  % Not use directly objGpet.nSubsets, because it canbe the case where there is a number of susbets configured but we still want to project the shile sinogram.
                        end
                        if strcmpi(objGpet.method, 'otf_siddon_cpu')
                            [m, structSizeSinogram] = ProjectMmr(x, objGpet.image_size.voxelSize_mm, objGpet.tempPath, objGpet.sinogram_size.span, numSubsets, subset_i, 0);
                        elseif strcmpi(objGpet.method, 'otf_siddon_gpu')
                            [m, structSizeSinogram] = ProjectMmr(x, objGpet.image_size.voxelSize_mm, objGpet.tempPath, objGpet.sinogram_size.span, numSubsets, subset_i, 1);
                        end
                    end
                else
                    % Select the subsets:
                    if nargin < 3
                        subset_i = [];
                        numSubsets = []; 
                    else
                        numSubsets = objGpet.nSubsets;  % Not use directly objGpet.nSubsets, because it canbe the case where there is a number of susbets configured but we still want to project the shile sinogram.
                    end
                    if strcmpi(objGpet.method, 'otf_siddon_cpu')
                        [m, structSizeSinogram] = ProjectMmr2d(x, objGpet.image_size.voxelSize_mm, objGpet.tempPath, numSubsets, subset_i, 0);
                    elseif strcmpi(objGpet.method, 'otf_siddon_gpu')
                        [m, structSizeSinogram] = ProjectMmr2d(x, objGpet.image_size.voxelSize_mm, objGpet.tempPath, numSubsets, subset_i, 1);
                    end
                end
            else
                error('unkown scanner')
            end
        end
        
        function x = PT(objGpet,m, subset_i)

            %full/sub-backprojection
            if nargin <3
                angles = 0:objGpet.sinogram_size.nAnglesBins-1;
                subset_i = [];
                numSubsets = []; 
            else
                angles = objGpet.sinogram_size.subsets(:,subset_i);
                numSubsets = objGpet.nSubsets;  % Not use directly objGpet.nSubsets, because it canbe the case where there is a number of susbets configured but we still want to project the shile sinogram.
            end
            % Check the image size:
            if size(m) == 1
                % Uniform image with x value:
                m = ones(objGpet.sinogram_size.matrixSize).*m;
            else
                sizeM = size(m);
                if numel(sizeM) == 2
                    sizeM = [size(m) 1];% 2d, I need to add the z size because
                    % matrixSize is a 3-elements vector.
                end
                if sizeM ~= objGpet.sinogram_size.matrixSize
                    warning('m: the input image has a different size to the matrix_size of the proejctor');
                end
            end
            if strcmpi(objGpet.scanner,'2D_radon')
                if strcmpi(objGpet.method, 'otf_matlab')
                    x = iradon(m,angles,'none',objGpet.image_size.matrixSize(1));
                else
                    error(sprintf('The method %s is not available for the scanner %s.', objGpet.method, objGpet.scanner));
                end
            elseif strcmpi(objGpet.scanner,'mMR')
                if strcmpi(objGpet.method, 'pre-computed_matlab') 
                     % G = mMR_backward(m,G,angles); using G.method 
                    x = iradon(m,angles,'none',objGpet.image_size.matrixSize(1));
                elseif strcmpi(objGpet.method, 'otf_siddon_cpu')
                    [x, pixelSize] = BackprojectMmr(m, objGpet.image_size.matrixSize, objGpet.image_size.voxelSize_mm, objGpet.tempPath, objGpet.sinogram_size.span, numSubsets, subset_i, 0);
                elseif strcmpi(objGpet.method, 'otf_siddon_gpu')
                    [x, pixelSize] = BackprojectMmr(m, objGpet.image_size.matrixSize, objGpet.image_size.voxelSize_mm, objGpet.tempPath, objGpet.sinogram_size.span, numSubsets, subset_i, 1);
                end
            elseif strcmpi(objGpet.scanner,'2D_mMR')   
                if strcmpi(objGpet.method, 'pre-computed_matlab') 
                     % G = mMR_backward(m,G,angles); using G.method 
                    x = iradon(m,angles,'none',objGpet.image_size.matrixSize(1));
                elseif strcmpi(objGpet.method, 'otf_siddon_cpu')
                    [x, pixelSize] = BackprojectMmr2d(m, objGpet.image_size.matrixSize, objGpet.image_size.voxelSize_mm, objGpet.tempPath,numSubsets, subset_i, 0);
                elseif strcmpi(objGpet.method, 'otf_siddon_gpu')
                    [x, pixelSize] = BackprojectMmr2d(m, objGpet.image_size.matrixSize, objGpet.image_size.voxelSize_mm, objGpet.tempPath, numSubsets, subset_i, 1);
                end
            else
                error('unkown scanner')
            end
            % PSF convolution
            if strcmpi(objGpet.PSF.type,'shift-invar')
                x = Gauss3DFilter(x, objGpet.image_size, objGpet.PSF.Width);
            else
                disp('todo: shift-var')
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
            objGpet.image_size.voxelSize_mm = [2.08626 2.08626];

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

        function  osem_subsets(objGpet, nsub,nAngles)
            if nsub==nAngles
                objGpet.sinogram_size.subsize = 1;
                objGpet.sinogram_size.subsets = 1:nAngles;
                s = subsets;
                st = 1 + bit_reverse(nsub);
                for i= 1:nsub
                    objGpet.sinogram_size.subsets(:,i) = s(:,st(i));
                end
                return
            end

            if rem(nAngles/nsub,2)~=0
                i = 1:nAngles/2;
                j = ~mod(nAngles/2./i,1);
                error(['Choose a valid subset: '  sprintf('%d ',i(j))])
            end

            objGpet.sinogram_size.subsize = nAngles /nsub;
            objGpet.sinogram_size.subsets = zeros(objGpet.sinogram_size.subsize, nsub);

             for j = 1:nsub
                 k = 0;
                 for i = j:nsub:nAngles/2
                     k = k+1;
                     objGpet.sinogram_size.subsets(k,j) = i;
                     objGpet.sinogram_size.subsets(k+objGpet.sinogram_size.subsize/2,j) = i+nAngles/2;
                 end
             end

            s = objGpet.sinogram_size.subsets;
            st = 1 + bit_reverse(nsub);
            for i= 1:nsub
                objGpet.sinogram_size.subsets(:,i) = s(:,st(i));
            end
        end
        
        function set_subsets(objGpet, numSubsets)
            % Update number of iteration to keep the subset%iterations
            % constant:
            objGpet.nIter = objGpet.nIter*objGpet.nSubsets/numSubsets;
            objGpet.nSubsets = numSubsets;
            
            objGpet.osem_subsets(objGpet.nSubsets, objGpet.sinogram_size.nAnglesBins);
        end
        
        function init_sinogram_size(objGpet, inSpan, numRings, maxRingDifference)
            objGpet.sinogram_size.span = inSpan;
            objGpet.sinogram_size.nRings = numRings;
            objGpet.sinogram_size.maxRingDifference = maxRingDifference;
            % Number of planes mashed in each plane of the sinogram:
            objGpet.sinogram_size.numPlanesMashed = [];

            % Number of planes in odd and even segments:
            numPlanesOdd = floor(objGpet.sinogram_size.span/2);
            numPlanesEven = ceil(objGpet.sinogram_size.span/2);
            % Ahora tengo que determinar la cantidad de segmentos y las diferencias
            % minimas y maximas en cada uno de ellos. Se que las diferencias maximas y
            % minimas del segmento cero es +-span/2:
            objGpet.sinogram_size.minRingDiffs = -floor(objGpet.sinogram_size.span/2);
            objGpet.sinogram_size.maxRingDiffs = floor(objGpet.sinogram_size.span/2);
            objGpet.sinogram_size.nSeg = 1;
            % Empiezo a ir agregando los segmentos hasta llegar numRings o a la máxima
            % diferencia entra anillos:
            while(abs(objGpet.sinogram_size.minRingDiffs(objGpet.sinogram_size.nSeg)) < objGpet.sinogram_size.maxRingDifference)  % El abs es porque voy a estar comparando el segmento con diferencias negativas.
                % Si no llegue a esa condición, tengo un segmento más hacia cada lado
                % primero hacia el positivo y luego negativo:
                objGpet.sinogram_size.nSeg = objGpet.sinogram_size.nSeg+1;
                % Si estoy en el primer segmento a agregar es -1, sino es -2 para ir
                % con el positivo:
                if objGpet.sinogram_size.nSeg == 2
                    objGpet.sinogram_size.minRingDiffs(objGpet.sinogram_size.nSeg) = objGpet.sinogram_size.minRingDiffs(objGpet.sinogram_size.nSeg-1) + objGpet.sinogram_size.span;
                    objGpet.sinogram_size.maxRingDiffs(objGpet.sinogram_size.nSeg) = objGpet.sinogram_size.maxRingDiffs(objGpet.sinogram_size.nSeg-1) + objGpet.sinogram_size.span;
                else
                    % Si me paso de la máxima difrencia de anillos la trunco:
                    if (objGpet.sinogram_size.maxRingDiffs(objGpet.sinogram_size.nSeg-2) + span) <= maxRingDifference
                        objGpet.sinogram_size.minRingDiffs(objGpet.sinogram_size.nSeg) = objGpet.sinogram_size.minRingDiffs(objGpet.sinogram_size.nSeg-2) + objGpet.sinogram_size.span;
                        objGpet.sinogram_size.maxRingDiffs(objGpet.sinogram_size.nSeg) = objGpet.sinogram_size.maxRingDiffs(objGpet.sinogram_size.nSeg-2) + objGpet.sinogram_size.span;
                    else
                        objGpet.sinogram_size.minRingDiffs(objGpet.sinogram_size.nSeg) = objGpet.sinogram_size.minRingDiffs(objGpet.sinogram_size.nSeg-2) + objGpet.sinogram_size.span;
                        objGpet.sinogram_size.maxRingDiffs(objGpet.sinogram_size.nSeg) = objGpet.sinogram_size.maxRingDifference;
                    end
                end
                % Ahora hacia el lado de las diferencias negativas:
                objGpet.sinogram_size.nSeg = objGpet.sinogram_size.nSeg+1;
                if (abs(objGpet.sinogram_size.minRingDiffs(objGpet.sinogram_size.nSeg-2) - objGpet.sinogram_size.span)) <= objGpet.sinogram_size.maxRingDifference
                    objGpet.sinogram_size.minRingDiffs(objGpet.sinogram_size.nSeg) = minRingDiffs(objGpet.sinogram_size.nSeg-2) - objGpet.sinogram_size.span;  % Acá siempre debo ir -2 no tengo problema con el primero.
                    objGpet.sinogram_size.maxRingDiffs(objGpet.sinogram_size.nSeg) = maxRingDiffs(objGpet.sinogram_size.nSeg-2) - objGpet.sinogram_size.span;  
                else
                    objGpet.sinogram_size.minRingDiffs(objGpet.sinogram_size.nSeg) = -objGpet.sinogram_size.maxRingDifference;  % Acá siempre debo ir -2 no tengo problema con el primero.
                    objGpet.sinogram_size.maxRingDiffs(objGpet.sinogram_size.nSeg) = objGpet.sinogram_size.maxRingDiffs(objGpet.sinogram_size.nSeg-2) - objGpet.sinogram_size.span;  
                end
            end

            % Ahora determino la cantidad de sinogramas por segmentos, recorriendo cada
            % segmento:
            objGpet.sinogram_size.nPlanesPerSeg = zeros(1,objGpet.sinogram_size.nSeg);

            for segment = 1 : objGpet.sinogram_size.nSeg
                % Por cada segmento, voy generando los sinogramas correspondientes y
                % contándolos, debería coincidir con los sinogramas para ese segmento: 
                numSinosThisSegment = 0;
                % Recorro todos los z1 para ir rellenando
                for z1 = 1 : (numRings*2)
                    numSinosZ1inSegment = 0;   % Cantidad de sinogramas para z1 en este segmento
                    % Recorro completamente z2 desde y me quedo con los que están entre
                    % minRingDiff y maxRingDiff. Se podría hacer sin recorrer todo el
                    % sinograma pero se complica un poco.
                    z1_aux = z1;    % z1_aux la uso para recorrer.
                    for z2 = 1 : numRings
                        % Ahora voy avanzando en los sinogramas correspondientes,
                        % disminuyendo z1 y aumentnado z2 hasta que la diferencia entre
                        % anillos llegue a maxRingDiff.
                        if ((z1_aux-z2)<=objGpet.sinogram_size.maxRingDiffs(segment))&&((z1_aux-z2)>=objGpet.sinogram_size.minRingDiffs(segment))
                            % Me asguro que esté dentro del tamaño del michelograma:
                            if(z1_aux>0)&&(z2>0)&&(z1_aux<=numRings)&&(z2<=numRings)
                                numSinosZ1inSegment = numSinosZ1inSegment + 1;
                            end
                        end
                        % Pase esta combinación de (z1,z2), paso a la próxima:
                        z1_aux = z1_aux - 1;
                    end
                    if(numSinosZ1inSegment>0)
                        objGpet.sinogram_size.numPlanesMashed = [objGpet.sinogram_size.numPlanesMashed numSinosZ1inSegment];
                        numSinosThisSegment = numSinosThisSegment + 1;
                    end
                end 
                % Guardo la cantidad de segmentos:
                objGpet.sinogram_size.nPlanesPerSeg(segment) = numSinosThisSegment;
            end
            objGpet.sinogram_size.nSinogramPlanes = sum(objGpet.sinogram_size.nPlanesPerSeg);
            objGpet.sinogram_size.matrixSize = [objGpet.sinogram_size.nRadialBins objGpet.sinogram_size.nAnglesBins objGpet.sinogram_size.nSinogramPlanes];
            
        end

        function objGpet=init_image_properties(objGpet, refImage)
            objGpet.image_size.matrixSize = refImage.ImageSize;
            objGpet.image_size.voxelSize_mm = [refImage.PixelExtentInWorldY refImage.PixelExtentInWorldX refImage.PixelExtentInWorldZ];
        end
    end
end







