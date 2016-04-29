% *********************************************************************
% Reconstruction Framework for Siemens Biograph mMR.  
% class: Gpet
% Authors: Martin Belzunce, Abolfazl Mehranian. Kings College London.
% Date: 08/02/2016
% *********************************************************************

function [n, n_ti, n_tv]=NCF(varargin)
     objGpet = varargin{1};
     if ~strcmpi(objGpet.scanner,'mMR') && ~strcmpi(objGpet.scanner,'2D_mMR')
         error('NCFs are only available for mMR scanner.');
     end
     if strcmpi(objGpet.scanner,'mMR')
         if nargin == 1
             % Default normalization file:
             [n, n_ti, n_tv, acquisition_dependant_ncf_3d, crystal_dependant_ncf_3d, used_xtal_efficiencies, used_deadtimefactors, used_axial_factors] = ...
                 create_norm_files_mmr([], [], [], [], [], objGpet.sinogram_size.span);
         elseif nargin == 2
             % Read it from file:
             [n, n_ti, n_tv, acquisition_dependant_ncf_3d, crystal_dependant_ncf_3d, used_xtal_efficiencies, used_deadtimefactors, used_axial_factors] = ...
                 create_norm_files_mmr(varargin{2}, [], [], [], [], objGpet.sinogram_size.span);
         end
     elseif strcmpi(objGpet.scanner,'2D_mMR')
         if nargin == 1
             % Default normalization file:
             [n, n_ti, n_tv, acquisition_dependant_ncf_3d, crystal_dependant_ncf_3d, used_xtal_efficiencies, used_deadtimefactors, used_axial_factors] = ...
                 create_norm_files_mmr_2d([], [], [], [], objGpet.sinogram_size.nRings);
         elseif nargin == 2
             % Read it from file:
             [n, n_ti, n_tv, acquisition_dependant_ncf_3d, crystal_dependant_ncf_3d, used_xtal_efficiencies, used_deadtimefactors, used_axial_factors] = ...
                 create_norm_files_mmr_2d(varargin{2}, [], [], [], objGpet.sinogram_size.nRings);
         end
     end
 end