%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 04/03/2015
%  *********************************************************************
%  Tries to estimate the crystal efficiencies from the measured data (suing
%  only direct sinogram).

clear all 
close all
%% PATHS FOR EXTERNAL FUNCTIONS AND RESULTS
addpath('/home/mab15/workspace/Biograph_mMr/mmr');
apirlPath = '/home/mab15/workspace/apirl-code/trunk/';
addpath(genpath([apirlPath '/matlab']));
setenv('PATH', [getenv('PATH') ':/home/mab15/workspace/apirl-code/trunk/build/bin']);
setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') ':/home/mab15/workspace/apirl-code/trunk/build/bin']);
outputPath = '/workspaces/Martin/KCL/Biograph_mMr/mmr/5hr_ge68/';
%setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') ':/usr/lib/x86_64-linux-gnu/']);
%% READING THE SINOGRAMS
% Read the sinograms:
filenameUncompressedMmr = '/home/mab15/workspace/KCL/Biograph_mMr/mmr/5hr_ge68/cylinder_5hours.s';
outFilenameIntfSinograms = '/home/mab15/workspace/KCL/Biograph_mMr/mmr/5hr_ge68/cylinder_5hoursIntf';
[sinogram, delayedSinogram, structSizeSino3d] = getIntfSinogramsFromUncompressedMmr(filenameUncompressedMmr, outFilenameIntfSinograms);

%% DIRECT SINOGRAMS
numSinos2d = structSizeSino3d.sinogramsPerSegment(1);
structSizeSino2d = getSizeSino2dStruct(structSizeSino3d.numTheta, structSizeSino3d.numR, ...
    numSinos2d, structSizeSino3d.rFov_mm, structSizeSino3d.zFov_mm);
directSinograms = single(sinogram(:,:,1:structSizeSino2d.numZ));
% Write to a file in interfile formar:
outputSinogramName = [outputPath '/directSinograms'];
interfileWriteSino(single(directSinograms), outputSinogramName);
%% NORMALIZATION
cbn_filename = '/home/mab15/workspace/KCL/Biograph_mMr/mmr/Norm_20141008101010.n';
[overall_ncf_3d, scanner_time_invariant_ncf_3d, scanner_time_variant_ncf_3d, acquisition_dependant_ncf_3d, used_xtal_efficiencies, used_deadtimefactors, used_axial_factors] = ...
   create_norm_files_mmr(cbn_filename, [], [], [], [], 1);
scanner_time_invariant_ncf_direct = scanner_time_invariant_ncf_3d(1:structSizeSino3d.numZ);
%% ATTENUATION CORRECTION - PICK A OR B AND COMMENT THE NOT USED 
%% COMPUTE THE ACFS (OPTION A)
% Read the phantom and then generate the ACFs with apirl:
imageSizeAtten_pixels = [344 344 127];
pixelSizeAtten_mm = [2.08626 2.08626 2.0312];
imageSizeAtten_mm = imageSizeAtten_pixels .* pixelSizeAtten_mm;
filenameAttenMap = '/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/2601/interfile/PET_ACQ_16_20150116131121-0_PRR_1000001_20150126152442_umap_human_00.v';
fid = fopen(filenameAttenMap, 'r');
if fid == -1
    ferror(fid);
end
attenMap = fread(fid, imageSize_pixels(1)*imageSize_pixels(2)*imageSize_pixels(3), 'single');
attenMap = reshape(attenMap, imageSize_pixels);
fclose(fid);
% visualization
figure;
image = getImageFromSlices(attenMap, 12, 1, 0);
imshow(image);

% 
% % Create ACFs of a computed phatoms with the linear attenuation
% % coefficients:
% Now the same for 2d sinograms:
acfFilename = ['acfsDirectSinograms'];
filenameSinogram = [outputPath 'directSinograms'];
acfsDirectSinograms = createACFsFromImage(attenMap, sizePixel_mm, outputPath, acfFilename, filenameSinogram, structSizeSino2d, 1);
%% READ THE ACFS (OPTION B)
acfFilename = [outputPath 'acfsDirectSinograms'];
% Direct Sinogram:
fid = fopen([acfFilename '.i33'], 'r');
[acfsDirectSinograms, count] = fread(fid, structSizeSino2d.numTheta*structSizeSino2d.numR*structSizeSino2d.numZ, 'single=>single');
acfsDirectSinograms = reshape(acfsDirectSinograms, [structSizeSino2d.numR structSizeSino2d.numTheta structSizeSino2d.numZ]);
% Close the file:
fclose(fid);
%% ATTENUATION FACTORS
afsDirectSinograms = acfsDirectSinograms;
afsDirectSinograms(afsDirectSinograms~=0) = 1./(afsDirectSinograms(afsDirectSinograms~=0));
%% ML-EM FOR NORMALIZATION
% em_recon = ones(nx, ny);
%              [  mask ] = make_cylinder( nx, ny, mm_x, 0, 0, mm_x*0.4*nx, 1.0,    0 );   % MU for GERMANIUM CYL  
% 
% em_recon = em_recon .* mask;
%              
% 
% af_sino = acf_sino;
% 
% nonzerosinoindices = find(acf_sino > 0.0);
% af_sino(nonzerosinoindices) =  1.0 ./ af_sino(nonzerosinoindices);
% 
% 
% sens_im = iradon(gaps_sinogram.*af_sino, phi_angles, 'none');
% 
%  if(visualize == 1)
%           figure(111);
%  end
% 
% for iter=1:its_2_do
%     
%   fp = radon(em_recon,phi_angles,num_rad_bins);
%   fp = fp.*af_sino;
%   if(visualize == 1)
%       subplot(3,3,1), display_image( em_recon, 'em_recon', 0.0 );
%       subplot(3,3,4), display_image( sens_im, 'sens_im', 0 );
%       subplot(3,3,2), display_image( measured_sinogram', 'measured_sinogram', 0 ); 
%   end
%   
%   nonzero = find(fp > 10);
%   sratio = measured_sinogram; sratio(:)=0.0;
%   sratio(nonzero) = measured_sinogram(nonzero) ./ fp(nonzero);
%   
%   if(visualize == 1)
%       subplot(3,3,5), display_image( fp', 'fp', 0 ); 
%       subplot(3,3,3), display_image( sratio', 'sratio', 0 ); 
%       subplot(3,3,6), display_image( af_sino', 'af_sino', 0 ); 
% 
%       drawnow
%       
%   end
%   
%   bp =   iradon(sratio.*af_sino,phi_angles,'none');
%  
%   em_recon = (em_recon ./ sens_im) .* bp;        
%   
%   em_recon = em_recon .* mask;
% 
% end