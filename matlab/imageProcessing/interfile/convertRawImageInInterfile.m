%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 06/05/2015
%  *********************************************************************
clear all
close all
filenameRawData = '/home/mab15/workspace/KCL/Phantoms/NCAT/NCATbin.i33';
sizeImage_pixels = [128 128 128];
sizePixels_mm = [3.125 3.125 3.125];
typeData = 'uint16';

fid = fopen(filenameRawData, 'r');
image = fread(fid, inf, typeData);
image = reshape(image, sizeImage_pixels);
image = permute(image, [2 1 3]);

outputFilename = '/home/mab15/workspace/KCL/Phantoms/NCAT/NCAT';
interfilewrite(image, outputFilename, sizePixels_mm);
