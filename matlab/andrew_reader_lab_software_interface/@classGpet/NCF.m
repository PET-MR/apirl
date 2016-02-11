% *********************************************************************
% Reconstruction Framework for Siemens Biograph mMR.  
% class: Gpet
% Authors: Martin Belzunce, Abolfazl Mehranian. Kings College London.
% Date: 08/02/2016
% *********************************************************************

function n=NCF(varargin)
     objGpet = varargin{1};
     if ~strcmpi(objGpet.scanner,'mMR')
         error('NCFs are only available for mMR scanner.');
     end
     if nargin == 1
         % Default normalization file:
         [overall_ncf_3d, scanner_time_invariant_ncf_3d, scanner_time_variant_ncf_3d, acquisition_dependant_ncf_3d, used_xtal_efficiencies, used_deadtimefactors, used_axial_factors] = ...
             create_norm_files_mmr([], [], [], [], [], objGpet.sinogram_size.span);
         n = overall_ncf_3d;
     elseif nargin == 2
         % Read it from file:
         [overall_ncf_3d, scanner_time_invariant_ncf_3d, scanner_time_variant_ncf_3d, acquisition_dependant_ncf_3d, used_xtal_efficiencies, used_deadtimefactors, used_axial_factors] = ...
             create_norm_files_mmr(varargin{2}, [], [], [], [], objGpet.sinogram_size.span);
         n = overall_ncf_3d;
     end
 end