%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 24/03/2015
%  *********************************************************************
%  This function get an spatially variant mean using a mask to choose the
%  ROI.

function [volume_filtered_masked maskEroded] = getSpatiallyVariantMean(volume, mask)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
SE = strel('disk',3);
maskEroded = imerode(mask, SE);

mask_scaled = mask .* mean(volume(mask));

% Filter for SNR:
filter = fspecial('gaussian',[21 21],7);
volume_filtered = zeros(size(volume));
for i = 1 : size(volume,3)
    volume_filtered(:,:,i) = imfilter(volume(:,:,i), filter);
    mask_filtered(:,:,i) =  imfilter(mask_scaled(:,:,i), filter);
end

volume_filtered_masked = zeros(size(mask));
volume_filtered_masked(mask_filtered ~= 0) = mean(volume(mask)) .* volume_filtered(mask_filtered ~= 0) ./ mask_filtered(mask_filtered ~= 0);
volume_filtered_masked = volume_filtered_masked .* maskEroded;



