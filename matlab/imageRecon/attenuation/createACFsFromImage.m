%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 27/01/2015
%  *********************************************************************
%  This function generates the attenuation correction factors (acfs) from
%  an attenuation map image, received as a paramete. The image it should be
%  in 1/cm. To achieve this, a configuration file is created to
%  generate the ACFs for different types of sinograms with APIRL, and the
%  run it.
%  It receives as a parameter the image with the attenuation map
%  an ouputpath, where not only stores the acfs
%  but also other intermediate files, and then the name for the specific
%  acfFile. It also receives the filename of the sinogram which is intenden
%  to correct for attenuation as it's needed in APIRl. And finally the
%  struct with the size of the sinogram.
function acfs = createACFsFromImage(attenuationMap_1_cm, sizePixel_mm, outputPath, acfFilename, filenameSinogram, structSizeSinos, visualization)
%% PATHS FOR EXTERNAL FUNCTIONS
addpath('/workspaces/Martin/KCL/Biograph_mMr/mmr');
apirlPath = '/workspaces/Martin/PET/apirl-code/trunk/';
addpath(genpath([apirlPath '/matlab']));

% Call function to create phantom:
attenuationMapFilename = [outputPath 'attenMapPhantom'];
% Write image to be able to read it from APIRL:
interfilewrite(single(attenuationMap_1_cm), attenuationMapFilename, sizePixel_mm);
% Now with the phantom, we create the configuration file for the command
% generateACFs:
genAcfFilename = [outputPath 'genACFs_' acfFilename '.par'];
% I have to add the extensions of the interfiles:
CreateGenAcfConfigFile(genAcfFilename, structSizeSinos, [filenameSinogram '.h33'], [attenuationMapFilename '.h33'], acfFilename);

% Then execute APIRL:
status = system(['generateACFs ' genAcfFilename]); 

% Read the generated acfs:
if isfield(structSizeSinos,'sinogramsPerSegment')
    numSinos = sum(structSizeSinos.sinogramsPerSegment);
else
    numSinos = structSizeSinos.numZ;
end

fid = fopen([acfFilename '.i33'], 'r');
[acfs, count] = fread(fid, structSizeSinos.numTheta*structSizeSinos.numR*numSinos, 'single=>single');
acfs = reshape(acfs, [structSizeSinos.numR structSizeSinos.numTheta numSinos]);
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