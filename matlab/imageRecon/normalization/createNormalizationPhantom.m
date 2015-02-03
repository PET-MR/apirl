%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 19/01/2015
%  *********************************************************************
%  This function creates a image of the Ge-68 normalization phantom and
%  write its in interfile format. 
function imageAtenuation = createNormalizationPhantom(sizeImage_pixels, sizeImage_mm, outputFilename, visualization)
%% PATHS FOR EXTERNAL FUNCTIONS
addpath('/workspaces/Martin/KCL/Biograph_mMr/mmr');
apirlPath = '/workspaces/Martin/PET/apirl-code/trunk/';
addpath(genpath([apirlPath '/matlab']));

%% OUTPUT PATH
% Outputh were images and other files are writen:
outputPath = '/workspaces/Martin/KCL/Biograph_mMr/mmr/5hr_ge68/';
%% INICIALIZACIÓN IMÁGENES
% Pixel size:
sizePixel_mm = sizeImage_mm ./ sizeImage_pixels;
imageAtenuation = zeros(sizeImage_pixels);
mu_mass_aire = 8.712E-02;
densidad_aire = 1.205E-03;
mu_lineal_aire_1_mm = mu_mass_aire *densidad_aire / 10;
imageAtenuation(:) = mu_lineal_aire_1_mm;
%% COORDINATE SYSTEM
% El x vanza como los índices, osea a la izquierda es menor, a la derecha
% mayor.
coordX = -((sizeImage_mm(2)/2)-sizePixel_mm(1)/2):sizePixel_mm(1):((sizeImage_mm(2)/2)-sizePixel_mm(1)/2);
% El y y el z van al revés que los índices, o sea el valor geométrico va a
% contramano con los índices de las matrices.
coordY = -((sizeImage_mm(1)/2)-sizePixel_mm(2)/2):sizePixel_mm(2):((sizeImage_mm(1)/2)-sizePixel_mm(2)/2);
coordZ = -((sizeImage_mm(3)/2)-sizePixel_mm(3)/2):sizePixel_mm(3):((sizeImage_mm(3)/2)-sizePixel_mm(3)/2);
[X,Y,Z] = meshgrid(coordX, coordY, coordZ);

%% PHANTOM
% Linear attenuation coefficent of Ge-68 resine:
% Mass attenuation coefficient of active resin@511 keV
% 0.103 (cm2/g)
mu_mass_ge = 1.03E-01;
densidad_ge = 1;
mu_ge_resine_1_mm = mu_mass_ge * densidad_ge / 10;
% Size of the phantom
radiusGeCylinder_mm = 100;
heightGeCylinder_mm = 275;
indexGeCylinder = (sqrt((X-0).^2+(Y-0).^2) < radiusGeCylinder_mm)  & (Z>-(heightGeCylinder_mm/2))&(Z<(heightGeCylinder_mm/2));
imageAtenuation(indexGeCylinder) = mu_ge_resine_1_mm;
%% VISUALIZATION
if visualization == 1
    image = getImageFromSlices(imageAtenuation, 12, 1, 0);
    figure;
    imshow(image);
end
%% ESCRITURA DE IMÁGENES EN INTERFILE
interfilewrite(single(imageAtenuation), outputFilename, sizePixel_mm);