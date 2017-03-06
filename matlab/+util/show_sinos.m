function [ res ] = show_sinos( set_of_sinograms, display_level, figure_title, fignum )
%UNTITLED Summary of this function goes here
%   display_level: 1,2 or 3: the number of orthogonal views

[radial_bins,azimuthal_bins,number_sinograms] = size(set_of_sinograms)
 
min_sinogram = zeros(radial_bins,azimuthal_bins);
max_sinogram = zeros(radial_bins,azimuthal_bins);
med_sinogram = zeros(radial_bins,azimuthal_bins);
mean_sinogram = zeros(radial_bins,azimuthal_bins);
stdv_sinogram = zeros(radial_bins,azimuthal_bins);
mode_sinogram = zeros(radial_bins,azimuthal_bins);
% 
% parfor rad=1:radial_bins
%     for azi=1:azimuthal_bins
%         data_vector = set_of_sinograms(rad,azi,:);
%         data_vector = data_vector(find(data_vector>0));
%         if(size(data_vector,3) > 0)
%         min_sinogram(rad,azi) =    min(data_vector(:));
%         max_sinogram(rad,azi) =    max(data_vector(:));
%         med_sinogram(rad,azi) = median(data_vector(:));
%         mean_sinogram(rad,azi) =  mean(data_vector(:));
%         stdv_sinogram(rad,azi) =   std(data_vector(:));
%         mode_sinogram(rad,azi) =  mode(data_vector(:));
%         end
%     end
% end
% Whith zeros:
min_sinogram =    double(min(set_of_sinograms, [], 3));
max_sinogram =    double(max(set_of_sinograms, [], 3));
med_sinogram= double(median(set_of_sinograms,3));
mean_sinogram =  double(mean(set_of_sinograms,3));
stdv_sinogram =   double(std(set_of_sinograms,0,3));
mode_sinogram =  double(mode(set_of_sinograms,3));

        
figure(fignum); axis off;
subplot(3,2,1), subimage(min_sinogram, [-0.001 max(min_sinogram(:))]); title([figure_title ':Min']);
subplot(3,2,2), subimage(max_sinogram, [-0.001 max(max_sinogram(:))]); title([figure_title ':Max']);
subplot(3,2,3), subimage(med_sinogram, [-0.001 max(med_sinogram(:))]); title([figure_title ':Median']);
subplot(3,2,4), subimage(mean_sinogram, [-0.001 max(mean_sinogram(:))]); title([figure_title ':Mean']);
subplot(3,2,5), subimage(stdv_sinogram, [-0.001 max(stdv_sinogram(:))]); title([figure_title ':Std']);
subplot(3,2,6), subimage(mode_sinogram, [-0.001 max(stdv_sinogram(:))]); title([figure_title ':Mode']);


if display_level > 1
    
% min_sinogram = zeros(azimuthal_bins,number_sinograms);
% max_sinogram = zeros(azimuthal_bins,number_sinograms);
% med_sinogram = zeros(azimuthal_bins,number_sinograms);
% mean_sinogram = zeros(azimuthal_bins,number_sinograms);
% stdv_sinogram = zeros(azimuthal_bins,number_sinograms);
% mode_sinogram = zeros(azimuthal_bins,number_sinograms);
% 
% parfor azi=1:azimuthal_bins
%     for sino=1:number_sinograms
%         data_vector = set_of_sinograms(:,azi,sino);
%         data_vector = data_vector(find(data_vector>0));
%         if(size(data_vector,1) > 0)
%           min_sinogram(azi,sino) = min(data_vector(:));
%           max_sinogram(azi,sino) = max(data_vector(:));
%           med_sinogram(azi,sino) = median(data_vector(:));
%           mean_sinogram(azi,sino) = mean(data_vector(:));
%           stdv_sinogram(azi,sino) = std(data_vector(:));
%           mode_sinogram(azi,sino) = mode(data_vector(:));
%         end
%     end
% end
permutedSinos = permute(set_of_sinograms, [2 3 1]);
min_sinogram =    double(min(permutedSinos, [], 3));
max_sinogram =    double(max(permutedSinos, [], 3));
med_sinogram= double(median(permutedSinos,3));
mean_sinogram =  double(mean(permutedSinos,3));
stdv_sinogram =   double(std(permutedSinos,0,3));
mode_sinogram =  double(mode(permutedSinos,3));

figure(fignum+1);
subplot(3,2,1), subimage(min_sinogram, [-0.001 max(min_sinogram(:))]); title([figure_title ':Min']);
subplot(3,2,2), subimage(max_sinogram, [-0.001 max(max_sinogram(:))]); title([figure_title ':Max']);
subplot(3,2,3), subimage(med_sinogram, [-0.001 max(med_sinogram(:))]); title([figure_title ':Median']);
subplot(3,2,4), subimage(mean_sinogram, [-0.001 max(mean_sinogram(:))]); title([figure_title ':Mean']);
subplot(3,2,5), subimage(stdv_sinogram, [-0.001 max(stdv_sinogram(:))]); title([figure_title ':Std']);
subplot(3,2,6), subimage(mode_sinogram, [-0.001 max(stdv_sinogram(:))]); title([figure_title ':Mode']);


    
end

if display_level > 2
    
% min_sinogram = zeros(radial_bins,number_sinograms);
% max_sinogram = zeros(radial_bins,number_sinograms);
% med_sinogram = zeros(radial_bins,number_sinograms);
% mean_sinogram = zeros(radial_bins,number_sinograms);
% stdv_sinogram = zeros(radial_bins,number_sinograms);
% mode_sinogram = zeros(radial_bins,number_sinograms);
% 
% parfor rad=1:radial_bins
%     for sino=1:number_sinograms
%         data_vector = set_of_sinograms(rad,:,sino);
%         data_vector = data_vector(find(data_vector>0));
%         if(size(data_vector,2) > 0)
%           min_sinogram(rad,sino) = min(data_vector(:));
%           max_sinogram(rad,sino) = max(data_vector(:));
%           med_sinogram(rad,sino) = median(data_vector(:));
%           mean_sinogram(rad,sino) = mean(data_vector(:));
%           stdv_sinogram(rad,sino) = std(data_vector(:));
%           mode_sinogram(rad,sino) = mode(data_vector(:));
%         end
%     end
% end

permutedSinos = permute(set_of_sinograms, [1 3 2]);
min_sinogram =    double(min(permutedSinos, [], 3));
max_sinogram =    double(max(permutedSinos, [], 3));
med_sinogram= double(median(permutedSinos,3));
mean_sinogram =  double(mean(permutedSinos,3));
stdv_sinogram =   double(std(permutedSinos,0,3));
mode_sinogram =  double(mode(permutedSinos,3));
figure(fignum+2);
subplot(3,2,1), subimage(min_sinogram, [-0.001 max(min_sinogram(:))]); title([figure_title ':Min']);
subplot(3,2,2), subimage(max_sinogram, [-0.001 max(max_sinogram(:))]); title([figure_title ':Max']);
subplot(3,2,3), subimage(med_sinogram, [-0.001 max(med_sinogram(:))]); title([figure_title ':Median']);
subplot(3,2,4), subimage(mean_sinogram, [-0.001 max(mean_sinogram(:))]); title([figure_title ':Mean']);
subplot(3,2,5), subimage(stdv_sinogram, [-0.001 max(stdv_sinogram(:))]); title([figure_title ':Std']);
subplot(3,2,6), subimage(mode_sinogram, [-0.001 max(stdv_sinogram(:))]); title([figure_title ':Mode']);


    
end


% crash
% % Cluster sinograms, use B3D to denoise
% 
% num_clusters = 5;
% 
% cluster_means = zeros(radial_bins,azimuthal_bins, num_clusters);
% 
% % Initialise means of clusters
% 
% for cluster=1:num_clusters
%     cluster_means(:,:,cluster)=set_of_sinograms(:,:, floor((cluster / num_clusters) * number_sinograms));
% end
% 
% % Display means
% figure(100); clf; plotnum=0;
% for cluster=1:num_clusters
%   sinogram_show = cluster_means(:,:,cluster);
%   
%   plotnum=plotnum+1;
%   subplot(3,3,plotnum), subimage(sinogram_show, [0 max(sinogram_show(:))]);
%    
% end
% crash
% 
% 
% plotnum=0;sinogram_just_shown=zeros(radial_bins,azimuthal_bins);
% figure(200); clf;
% sinogram_show
% for sino=100:199
%   sinogram_show = double(set_of_sinograms(:,:,sino));
%   
%   plotnum=plotnum+1;
%   subplot(10,10,plotnum), subimage(sinogram_show, [0 max(sinogram_show(:))]);
%    
% end
res = 1;

end

