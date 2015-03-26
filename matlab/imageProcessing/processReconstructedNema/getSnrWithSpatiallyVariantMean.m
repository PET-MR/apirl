%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 24/03/2015
%  *********************************************************************
%  This function creates SNR for an image with a bias problem (for example
%  without scatter correction).

function [snr snr_per_slice meanValue mean_per_slice stdValue std_per_slice] = getSnrWithSpatiallyVariantMean(volume, mask)
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

% Get snr:
meanValue = mean(volume_filtered_masked(maskEroded));
stdValue = sqrt(mean((volume(maskEroded) - volume_filtered_masked(maskEroded)).^2));
snr = meanValue ./ stdValue;

snr_per_slice = zeros(size(volume,3),1);
for i = 1 : size(volume,3)
    if(sum(sum(mask(:,:,i))) ~= 0)
        slice = volume(:,:,i);
        sliceFiltered = volume_filtered_masked(:,:,i);
        mean_per_slice(i) = mean(sliceFiltered(maskEroded(:,:,i)));
        std_per_slice(i) = sqrt(mean((slice(maskEroded(:,:,i)) - mean_per_slice(i)).^2));
        snr_per_slice(i) = mean_per_slice(i) / std_per_slice(i);
    else
        snr_per_slice(i) = 0;
    end
end

end

