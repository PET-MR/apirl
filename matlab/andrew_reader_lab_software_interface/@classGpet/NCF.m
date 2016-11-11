% *********************************************************************
% Reconstruction Framework for Siemens Biograph mMR.  
% class: Gpet
% Authors: Martin Belzunce, Abolfazl Mehranian. Kings College London.
% Date: 08/02/2016
% *********************************************************************

function [n, n_ti, n_tv]=NCF(varargin)
     objGpet = varargin{1};
     if ~strcmpi(objGpet.scanner,'mMR')
         error('NCFs are only available for mMR scanner.');
     end
     if objGpet.sinogram_size.span >= 1
         if nargin == 1
             % Default normalization file:
             [n, n_ti, n_tv, acquisition_dependant_ncf_3d, crystal_dependant_ncf_3d, used_xtal_efficiencies, used_deadtimefactors, used_axial_factors] = ...
                 create_norm_files_mmr([], [], [], [], [], objGpet.sinogram_size.span);
         elseif nargin == 2
             % Read it from file:
             [~,name]= fileparts(varargin{2});
             if objGpet.sinogram_size.span==11 && strcmpi(name,'norm3d_00') % read ncf from e7 output
                 fid = fopen(varargin{2},'r'); n = fread(fid,inf,'float32'); fclose(fid);
                 n = reshape(n,objGpet.sinogram_size.nRadialBins,objGpet.sinogram_size.nAnglesBins,objGpet.sinogram_size.nSinogramPlanes);
             else
             
             [n, n_ti, n_tv, acquisition_dependant_ncf_3d, crystal_dependant_ncf_3d, used_xtal_efficiencies, used_deadtimefactors, used_axial_factors] = ...
                 create_norm_files_mmr(varargin{2}, [], [], [], [], objGpet.sinogram_size.span);
             end
         end
     else
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