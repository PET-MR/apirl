% *********************************************************************
% Reconstruction Framework for Siemens Biograph mMR.  
% class: Gpet
% Authors: Martin Belzunce, Abolfazl Mehranian. Kings College London.
% Date: 08/02/2016
% *********************************************************************
% Computes the scatter using one of the availables methods.
function s=S(varargin)
    objGpet = varargin{1};
    h = fspecial('gaussian',30,10);
    s = [];
    if ~strcmpi(objGpet.scanner,'mMR')
        error('NCFs are only available for mMR scanner.');
    end
    if nargin == 2 % Simple simulation, smooth a sinogram.
        s = zeros(size(varargin{2}));
        for i = 1 : size(s,3)
            s(:,:,i) = imfilter(varargin{2}(:,:,i), h, 'same');
        end
        % Fit tail?
    elseif nargin == 3 % Sss simulation, need activty image and attenuation.
        if strcmpi(objGpet.scatter_algorithm,'e7_tools')
            % Call e7 tools:
        elseif strcmpi(objGpet.scatter_algorithm,'sss_stir')
            % Call stir:
        end
    end
 end