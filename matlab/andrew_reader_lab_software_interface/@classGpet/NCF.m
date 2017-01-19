% *********************************************************************
% Reconstruction Framework for Siemens Biograph mMR.  
% class: Gpet
% Authors: Martin Belzunce, Abolfazl Mehranian. Kings College London.
% Date: 08/02/2016
% *********************************************************************

function [n, n_ti, n_tv, gaps]=NCF(varargin)
    n = []; n_ti = []; n_tv = [];
    objGpet = varargin{1};
    if nargin == 3 
        singles_per_bucket = varargin{3};
    else
        singles_per_bucket = [];
    end
    if ~strcmpi(objGpet.scanner,'mMR')
        error('NCFs are only available for mMR scanner.');
    end
    if strcmp(objGpet.method_for_normalization, 'from_e7_binary_interfile')
        if nargin < 2
            error('Method ''from_e7_binary_interfile'' requires the binary file as input parameter. Consider PET.method_for_normalization = ''cbn_expansion'' ');
        end
        if objGpet.sinogram_size.span == 11
            fid = fopen(varargin{2},'r');
            n = fread(fid,inf,'float32');
            fclose(fid);
            n = reshape(n,objGpet.sinogram_size.nRadialBins,objGpet.sinogram_size.nAnglesBins,objGpet.sinogram_size.nSinogramPlanes);
        else
            error('e7 supports only span 11. Consider PET.method_for_normalization = ''cbn_expansion'' ');
        end
    end
    if strcmp(objGpet.method_for_normalization, 'cbn_expansion')
        if objGpet.sinogram_size.span >= 1
            if nargin == 1
                % Default normalization file:
                [n, n_ti, n_tv, acquisition_dependant_ncf_3d, crystal_dependant_ncf_3d, gaps, used_xtal_efficiencies, used_deadtimefactors, used_axial_factors, structSizeSino3d] = ...
                    create_norm_files_mmr([], [], [], [], [], objGpet.sinogram_size.span);
            else
                [n, n_ti, n_tv, acquisition_dependant_ncf_3d, crystal_dependant_ncf_3d, gaps, used_xtal_efficiencies, used_deadtimefactors, used_axial_factors, structSizeSino3d] = ...
                create_norm_files_mmr(varargin{2}, [], [], [], singles_per_bucket, objGpet.sinogram_size.span);
            end
        else
            if nargin == 1
                % Default normalization file:
                [n, n_ti, n_tv, acquisition_dependant_ncf_3d, crystal_dependant_ncf_3d, used_xtal_efficiencies, used_deadtimefactors, used_axial_factors] = ...
                create_norm_files_mmr_2d([], [], [], [], objGpet.sinogram_size.nRings);
            else
                % Read it from file:
                [n, n_ti, n_tv, acquisition_dependant_ncf_3d, crystal_dependant_ncf_3d, used_xtal_efficiencies, used_deadtimefactors, used_axial_factors] = ...
                create_norm_files_mmr_2d(varargin{2}, [], [], singles_per_bucket, objGpet.sinogram_size.nRings);
            end
        end
    end
end
