%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 19/01/2015
%  *********************************************************************
%  This scripts creates a image of the Ge-68 normalization phantom and
%  write its in interfile format. It also generates a configuration file to
%  generate the ACFs for different types of sinograms with APIRL, and the
%  run it.
%  It receives as a parameter an ouputpath, where not only stores the acfs
%  but also other intermediate files, and then the name for the specific
%  acfFile. It also receives the filename of the sinogram which is intenden
%  to correct for attenuation as it's needed in APIRl. And finally the
%  struct with the size of the sinogram.
function acfs = createNormalizationPhantomACFs(outputPath, acfFilename, filenameSinogram, structSizeSinos, visualization)
%% PATHS FOR EXTERNAL FUNCTIONS
addpath('/workspaces/Martin/KCL/Biograph_mMr/mmr');
apirlPath = '/workspaces/Martin/PET/apirl-code/trunk/';
addpath(genpath([apirlPath '/matlab']));

%% IMAGE SIZES
% Size of the image to cover the full fov:
sizeImage_mm = [structSizeSinos.rFov_mm*2 structSizeSinos.rFov_mm*2 structSizeSinos.zFov_mm];
% The size in pixels based in numR and the number of rings:
sizeImage_pixels = [structSizeSinos.numR structSizeSinos.numR structSizeSinos.numZ];
% Call function to create phantom:
attenuationMapFilename = [outputPath '/normalizationPhantom'];
imageAtenuation = createNormalizationPhantom(sizeImage_pixels, sizeImage_mm, attenuationMapFilename, visualization);

% Now with the phantom, we create the configuration file for the command
% generateACFs:
genAcfFilename = [outputPath '/genACFs_' acfFilename '.par'];
% I have to add the extensions of the interfiles:
CreateGenAcfConfigFile(genAcfFilename, 'Sinograms2D', [filenameSinogram '.h33'], [attenuationMapFilename '.h33'], acfFilename);

% Then execute APIRL:
status = system(['generateACFs ' genAcfFilename]); 

% Read the generated acfs:
fid = fopen([acfFilename '.i33'], 'r');
[acfs, count] = fread(fid, structSizeSinos.numTheta*structSizeSinos.numR*structSizeSinos.numZ, 'single=>single');
acfs = reshape(acfs, [structSizeSinos.numR structSizeSinos.numTheta structSizeSinos.numZ]);
% Matlab reads in a column-wise order that why angles are in the columns.
% We want to have it in the rows since APIRL and STIR and other libraries
% use row-wise order:
acfs = permute(acfs,[2 1 3]);
% Close the file:
fclose(fid);

if visualization == 1
    image = getImageFromSlices(acfs,12);
    figure;
    imshow(image);
end