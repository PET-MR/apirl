%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 14/02/2015
%  *********************************************************************
%  This scripts analyze and verifies the normalization factors. It compares
%  with the ones that stir computes.
clear all 
close all
%% PATHS FOR EXTERNAL FUNCTIONS AND RESULTS
addpath('/workspaces/Martin/KCL/Biograph_mMr/mmr');
apirlPath = '/workspaces/Martin/PET/apirl-code/trunk/';
addpath(genpath([apirlPath '/matlab']));
setenv('PATH', [getenv('PATH') ':/workspaces/Martin/PET/apirl-code/trunk/build/bin']);
setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') ':/workspaces/Martin/PET/apirl-code/trunk/build//bin']);
outputPath = '/workspaces/Martin/KCL/Biograph_mMr/mmr/5hr_ge68/';
%setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') ':/usr/lib/x86_64-linux-gnu/']);
%% READ NORMALIZATION WITH OUR FUNCTION
cbn_filename = '/workspaces/Martin/STIR/KCL/STIR_mMR_KCL/IM/NORM.n';
[overall_ncf_3d, scanner_time_invariant_ncf_3d, scanner_time_variant_ncf_3d, acquisition_dependant_ncf_3d, used_xtal_efficiencies, used_deadtimefactors, used_axial_factors] = ...
   create_norm_files_mmr(cbn_filename, [], [], [], [], 11);



%% READ FACTORS FROM STIR
stirNormPath = '/workspaces/Martin/STIR/KCL/STIR_mMR_KCL/';

% STIR save in a differente format, so then we need to extract the
% sinograms per segment, read each segment and rearrange the whole
% sinogram:
fid = fopen('/workspaces/Martin/STIR/KCL/STIR_mMR_KCL/IM/NORMseg0_by_sino.v','r');
norm_factors_segment0 = fread(fid, 344*252*127, 'single');
fclose(fid);
norm_factors_segment0 = reshape(norm_factors_segment0, [344 252 127]);
% Here gaps are with a very high value:
norm_factors_segment0(norm_factors_segment0>100000) = 0;

%% Visualize two sinograms:
figure;
subplot(1,2,1);
imshow(overall_ncf_3d(:,:,1));
subplot(1,2,2);
imshow(norm_factors_segment0(:,:,1));

%% COMPARISON
for i = 1 : 127
    compSino2d =(overall_ncf_3d(:,:,i) >= norm_factors_segment0(:,:,i)*0.999)&(overall_ncf_3d(:,:,i) <= norm_factors_segment0(:,:,i)*1.001);
    if(sum(compSino2d(:)) ~= numel(compSino2d))
        disp(sprintf('Se detectaron eficiencias diferentes en el sinograma %d del segmento 0.\n',i));
        figure;imshow(~compSino2d);title(sprintf('Valores de Eficiencia Distintos para Sinograma %d',i));
    end
end
