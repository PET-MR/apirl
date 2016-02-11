% *********************************************************************
% Reconstruction Framework for Siemens Biograph mMR.  
% class: Gpet
% Authors: Martin Belzunce, Abolfazl Mehranian. Kings College London.
% Date: 08/02/2016
% *********************************************************************
% Computes the scatter using one of the availables methods.
function s=S(varargin)
    objGpet = varargin{1};
    if ~strcmpi(objGpet.scanner,'mMR')
        error('NCFs are only available for mMR scanner.');
    end
    if strcmpi(objGpet.scatter_algorithm,'e7_tools')
        % Call e7 tools:
    elseif strcmpi(objGpet.scatter_algorithm,'sss_stir')
        % Call stir:
    end
 end