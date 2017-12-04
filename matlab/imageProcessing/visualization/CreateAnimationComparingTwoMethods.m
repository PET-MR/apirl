function CreateAnimationComparingTwoMethods(imagesMethod1, imagesMethod2, params)
%% GENERATE THE TWO IMAGES
% Save all the frames:
slice_mlem = round(slice./size(nonlocal_lange_bowsher_mr_voxels.images{i},3).*size(mlem.images{1},3));
sizeSlice = size(nonlocal_lange_bowsher_mr_voxels.images{1}(:,:,slice));
indices_rows = 200:480;
indices_cols = 200:480; % To match the 235 slices
clear frames_both_recon
clear frames_both_with_labels
for i = iteration% 1 : numel(nonlocal_lange_bowsher_mr_voxels.images)
    frames_both_recon(:,:,i) = [mlem_resampled{i}(indices_rows,indices_cols,slice) nonlocal_lange_bowsher_mr_voxels.images{i}(:,:,slice)];
    frames_both_with_labels(:,:,:,i) = insertText(frames_both_recon(:,:,i), [60 numel(indices_rows)-35], 'Standard', 'BoxOpacity', 0.0, 'FontSize', 18, 'TextColor', 'white', 'BoxColor', 'yellow');%, 'Font', 'Arial');
    aux = rgb2gray(frames_both_with_labels(:,:,:,i));
    aux = insertText(aux, [60+numel(indices_cols) numel(indices_rows)-35], 'Proposed', 'BoxOpacity', 0.0, 'FontSize', 18, 'TextColor', 'white', 'BoxColor', 'yellow');%, 'Font', 'Arial');
    aux = rgb2gray(aux);
    aux = insertText(aux, [numel(indices_cols)-75 1], 'Reconstruction', 'BoxOpacity', 0.0, 'FontSize', 18, 'TextColor', 'white', 'BoxColor', 'yellow');%, 'Font', 'Arial');
    frames_both_recon(:,:,i) = rgb2gray(aux);
end
lastFrame = i;
% After that sweep all images:
slicesToShow = 25:100;
clear frames_slices
for i = 1 : numel(slicesToShow)% + numel(%size(mlem_resampled{1},3) % 1 lap and a half
    slice_i = rem(slice + i, slicesToShow(end))+floor((slice+i)/slicesToShow(end))*25;  
    % FIRST RESIZE THE SMALLER MLEM IMAGE
    frames_slices(:,:,i) = [mlem_resampled{iteration(end)}(indices_rows,indices_cols,slice_i) nonlocal_lange_bowsher_mr_voxels.images{iteration(end)}(:,:,slice_i)];
    frames_both_with_labels = insertText(frames_slices(:,:,i), [60 numel(indices_rows)-35], 'Standard', 'BoxOpacity', 0.0, 'FontSize', 18, 'TextColor', 'white', 'BoxColor', 'yellow');%, 'Font', 'Arial');
    aux = rgb2gray(frames_both_with_labels);
    aux = insertText(aux, [60+numel(indices_cols) numel(indices_rows)-35], 'Proposed', 'BoxOpacity', 0.0, 'FontSize', 18, 'TextColor', 'white', 'BoxColor', 'yellow');%, 'Font', 'Arial');
    aux = rgb2gray(aux);
    aux = insertText(aux, [numel(indices_cols)-60 1], 'Final Images', 'BoxOpacity', 0.0, 'FontSize', 18, 'TextColor', 'white', 'BoxColor', 'yellow');%, 'Font', 'Arial');
    frames_slices(:,:,i) = rgb2gray(aux);
end

clear frames_slices_in_order
for i = 1 : size(mlem_resampled{1},3) 
    slice_i = i;%rem(slice + i, size(mlem_resampled{1},3))+1;  
    % FIRST RESIZE THE SMALLER MLEM IMAGE
    frames_slices_in_order(:,:,i) = [mlem_resampled{iteration(end)}(indices_rows,indices_cols,slice_i) nonlocal_lange_bowsher_mr_voxels.images{iteration(end)}(:,:,slice_i)];
    frames_both_with_labels = insertText(frames_slices_in_order(:,:,i), [60 numel(indices_rows)-35], 'Standard', 'BoxOpacity', 0.0, 'FontSize', 18, 'TextColor', 'white', 'BoxColor', 'yellow');%, 'Font', 'Arial');
    aux = rgb2gray(frames_both_with_labels);
    aux = insertText(aux, [60+numel(indices_cols) numel(indices_rows)-35], 'Proposed', 'BoxOpacity', 0.0, 'FontSize', 18, 'TextColor', 'white', 'BoxColor', 'yellow');%, 'Font', 'Arial');
    aux = rgb2gray(aux);
    aux = insertText(aux, [numel(indices_cols)-60 1], 'Final Images', 'BoxOpacity', 0.0, 'FontSize', 18, 'TextColor', 'white', 'BoxColor', 'yellow');%, 'Font', 'Arial');
    frames_slices_in_order(:,:,i) = rgb2gray(aux);
end

filename = [outputPath sprintf('mlem_and_nonlocal_lange_bowsher_mr_res_slice%d_gray.gif',slice)];
writeAnimatedGif(frames_both_recon, filename, 0.1,gray, 0.5);

gray = colormap(gray);
invgray = gray(end:-1:1,:);
filename = [outputPath sprintf('mlem_and_nonlocal_lange_bowsher_mr_res_slice%d_invgray.gif',slice)];
writeAnimatedGif(frames_both_recon, filename, 0.1,invgray, 0.5);

filename = [outputPath sprintf('mlem_and_nonlocal_lange_bowsher_mr_res_slice%d_nih.gif',slice)];
writeAnimatedGif(frames_both_recon, filename, 0.1,nihMap, 0.5);

filename = [outputPath sprintf('mlem_and_nonlocal_lange_bowsher_mr_res_slice%d_hot.gif',slice)];
writeAnimatedGif(frames_both_recon, filename, 0.1,hot, 0.5);

filename = [outputPath 'mlem_and_nonlocal_lange_bowsher_mr_res_slices_gray.gif'];
writeAnimatedGif(frames_slices, filename, 0.1,gray, 0.5);

gray = colormap(gray);
invgray = gray(end:-1:1,:);
filename = [outputPath 'mlem_and_nonlocal_lange_bowsher_mr_res_slices_invgray.gif'];
writeAnimatedGif(frames_slices, filename, 0.1,invgray, 0.5);

filename = [outputPath 'mlem_and_nonlocal_lange_bowsher_mr_res_slices_nih.gif'];
writeAnimatedGif(frames_slices, filename, 0.1,nihMap, 0.5);

filename = [outputPath 'mlem_and_nonlocal_lange_bowsher_mr_res_slices_hot.gif'];
writeAnimatedGif(frames_slices, filename, 0.1,hot, 0.5);

%% GENERATE THE ROTATING MIPs
% Save all the frames:
slice = 80;
slice_mlem = round(slice./size(nonlocal_lange_bowsher_mr_voxels.images{i},3).*size(mlem.images{1},3));
sizeSlice = size(nonlocal_lange_bowsher_mr_voxels.images{1}(:,:,slice));
iteration = [1:19 20:2:28 30:4:70 80:10:150 160:20:300];
indices_rows_low_res = round(indices_rows./size(nonlocal_lange_bowsher_mr_voxels.images{i},1).*size(mlem.images{1},1));
indices_cols_low_res = round(indices_cols./size(nonlocal_lange_bowsher_mr_voxels.images{i},2).*size(mlem.images{1},2));
clear frames_both_rotating
angles = 0:3:357;
for i = 1 : numel(angles)
    [mipTransverseMlem, mipCoronalMlem, mipSagitalMlem] = getMIPs(mlem_resampled{iteration(end)}(indices_rows,indices_cols,:), angles(i));
    [mipTransverse, mipCoronal, mipSagital] = getMIPs(nonlocal_lange_bowsher_mr_voxels.images{iteration(end)}, angles(i));
    % Fill the coronal to math the size:
    additional_cols = size(frames_slices,1) - size(mipCoronal,1);
    mipCoronalMlem_filled = [zeros(round(additional_cols/2), size(mipCoronal,2)); mipCoronalMlem; zeros(round(additional_cols/2), size(mipCoronal,2))];
    mipCoronal_filled = [zeros(round(additional_cols/2), size(mipCoronal,2)); mipCoronal; zeros(round(additional_cols/2), size(mipCoronal,2))];
    frames_both_rotating(:,:,i) = [mipCoronalMlem_filled mipCoronal_filled]; % Crop to match the size of the pther images
    aux = insertText(frames_both_rotating(:,:,i), [60 size(frames_both_rotating,1)-35], 'Standard', 'BoxOpacity', 0.0, 'FontSize', 18, 'TextColor', 'white', 'BoxColor', 'yellow');%, 'Font', 'Arial');
    aux = rgb2gray(aux);
    aux = insertText(aux, [60+numel(indices_cols) size(frames_both_rotating,1)-35], 'Proposed', 'BoxOpacity', 0.0, 'FontSize', 18, 'TextColor', 'white', 'BoxColor', 'yellow');%, 'Font', 'Arial');
    aux = rgb2gray(aux);
    aux = insertText(aux, [numel(indices_cols)-75 1], '3D Volume', 'BoxOpacity', 0.0, 'FontSize', 18, 'TextColor', 'white', 'BoxColor', 'yellow');%, 'Font', 'Arial');
    frames_both_rotating(:,:,i) = rgb2gray(aux);
end
% for i = iteration% 1 : numel(nonlocal_lange_bowsher_mr_voxels.images)
%     % FIRST RESIZE THE SMALLER MLEM IMAGE
%     mlem_slice_resized = imresize(mlem.images{i}(:,:,slice_mlem), sizeSlice);
%     frames_both(:,:,i) = [mlem_slice_resized(indices_rows,indices_cols) nonlocal_lange_bowsher_mr_voxels.images{i}(indices_rows,indices_cols,slice)];
%     frames_both_with_labels(:,:,:,i) = insertText(frames_both(:,:,i), [60 numel(indices_rows)-35], 'Standard', 'BoxOpacity', 0.0, 'FontSize', 18, 'TextColor', 'white', 'BoxColor', 'yellow');%, 'Font', 'Arial');
%     aux = rgb2gray(frames_both_with_labels(:,:,:,i));
%     aux = insertText(aux, [60+numel(indices_cols) numel(indices_rows)-35], 'Proposed', 'BoxOpacity', 0.0, 'FontSize', 18, 'TextColor', 'white', 'BoxColor', 'yellow');%, 'Font', 'Arial');
%     aux = rgb2gray(aux);
%     aux = insertText(aux, [numel(indices_cols)-75 1], 'Reconstruction', 'BoxOpacity', 0.0, 'FontSize', 18, 'TextColor', 'white', 'BoxColor', 'yellow');%, 'Font', 'Arial');
%     frames_both(:,:,i) = rgb2gray(aux);
% end
% lastFrame = i;
% % After that sweep all images:
% slicesToShow = 25:100;
% for i = 1 : size(mlem.images{1},3)
%     slice_low_res = rem(slice_mlem + i, size(mlem.images{1},3))+1;  
%     slice_high_res = round(slice_low_res.*size(nonlocal_lange_bowsher_mr_voxels.images{i},3)./size(mlem.images{1},3));
%     % FIRST RESIZE THE SMALLER MLEM IMAGE
%     mlem_slice_resized = imresize(mlem.images{iteration(end)}(:,:,slice_low_res), sizeSlice);
%     frames_both(:,:,i+lastFrame) = [mlem_slice_resized(indices_rows,indices_cols) nonlocal_lange_bowsher_mr_voxels.images{iteration(end)}(indices_rows,indices_cols,slice_high_res)];
%     frames_both_with_labels(:,:,:,i+lastFrame) = insertText(frames_both(:,:,i+lastFrame), [60 numel(indices_rows)-35], 'Standard', 'BoxOpacity', 0.0, 'FontSize', 18, 'TextColor', 'white', 'BoxColor', 'yellow');%, 'Font', 'Arial');
%     aux = rgb2gray(frames_both_with_labels(:,:,:,i+lastFrame));
%     aux = insertText(aux, [60+numel(indices_cols) numel(indices_rows)-35], 'Proposed', 'BoxOpacity', 0.0, 'FontSize', 18, 'TextColor', 'white', 'BoxColor', 'yellow');%, 'Font', 'Arial');
%     aux = rgb2gray(aux);
%     aux = insertText(aux, [numel(indices_cols)-75 1], 'Final Images', 'BoxOpacity', 0.0, 'FontSize', 18, 'TextColor', 'white', 'BoxColor', 'yellow');%, 'Font', 'Arial');
%     frames_both(:,:,i+lastFrame) = rgb2gray(aux);
% end
%%
frames_both = cat(3, frames_both_recon, frames_slices, frames_both_rotating);
factors = [0.1:0.1:1 ones(1,size(frames_both_rotating,3)-numel([0.1:0.1:1]))];
for i = 1 : size(frames_both_rotating,3)
    frames_both_rotating_soft = frames_both_rotating*factors(i);
end
frames_both = cat(3, frames_both_recon, frames_slices, frames_both_rotating_soft);
frames_both_2 = cat(3, frames_both_recon, frames_both_rotating, frames_slices_in_order);

% Variation of nih colormap
logNihMap = log(1+nihMap)./log(2);
expNihMap = nihMap.^2;
nihMap_stretch_red = interp1(nihMap(200:end,2),nihMap(200:end,2), [1:-0.01:0]', 'linear','extrap'); nihMap_stretch_red = [repmat(1, size(nihMap_stretch_red)) nihMap_stretch_red repmat(0, size(nihMap_stretch_red))];
nihMap_stretch_yellow = interp1(nihMap(143:199,1),nihMap(143:199,1), [0:0.04:1]', 'linear','extrap');  nihMap_stretch_yellow = [ nihMap_stretch_yellow repmat(1, size(nihMap_stretch_yellow)) repmat(0, size(nihMap_stretch_yellow))];
nihMap_stretch_green = interp1(nihMap(87:142,3),nihMap(87:142,3), [0:0.08:1]', 'linear','extrap'); nihMap_stretch_green = [repmat(0, size(nihMap_stretch_green)) repmat(1, size(nihMap_stretch_green)) nihMap_stretch_green ];
nihMap_stretch_blue = interp1(nihMap(30:86,2),nihMap(30:86,2), [0:0.05:1]', 'linear','extrap'); nihMap_stretch_blue = [repmat(0, size(nihMap_stretch_blue)) nihMap_stretch_blue repmat(1, size(nihMap_stretch_blue)) ];
nihMap_extended = [nihMap(1:29,:); nihMap_stretch_blue; nihMap_stretch_green; nihMap_stretch_yellow; nihMap_stretch_red];
nihMap_extended(nihMap_extended<0) = 0;

scale = 1;
filename = [outputPath 'mlem_and_nonlocal_lange_bowsher_mr_res_seq_gray.gif'];
writeAnimatedGif(frames_both, filename, 0.1,gray, scale);

gray = colormap(gray);
invgray = gray(end:-1:1,:);
filename = [outputPath 'mlem_and_nonlocal_lange_bowsher_mr_res_seq_invgray.gif'];
writeAnimatedGif(frames_both, filename, 0.1,invgray, scale);


filename = [outputPath 'mlem_and_nonlocal_lange_bowsher_mr_res_seq_nih.gif'];
writeAnimatedGif(frames_both, filename, 0.1,nihMap_extended, scale);

filename = [outputPath 'mlem_and_nonlocal_lange_bowsher_mr_res_seq_hot.gif'];
writeAnimatedGif(frames_both, filename, 0.1,hot, scale);

filename = [outputPath 'mlem_and_nonlocal_lange_bowsher_mr_res_seq2_gray.gif'];
writeAnimatedGif(frames_both_2, filename, 0.1,gray, scale);

gray = colormap(gray);
invgray = gray(end:-1:1,:);
filename = [outputPath 'mlem_and_nonlocal_lange_bowsher_mr_res_seq2_invgray.gif'];
writeAnimatedGif(frames_both_2, filename, 0.1,invgray, scale);


filename = [outputPath 'mlem_and_nonlocal_lange_bowsher_mr_res_seq2_nih.gif'];
writeAnimatedGif(frames_both_2, filename, 0.1,nihMap, scale*0.9);

filename = [outputPath 'mlem_and_nonlocal_lange_bowsher_mr_res_seq2_hot.gif'];
writeAnimatedGif(frames_both_2, filename, 0.1,hot, scale);
%% WITH MR?
mrSlices = MrInPetFov(indices_rows,indices_cols,slice)./max(max(MrInPetFov(indices_rows,indices_cols,slice)));
figure, 
for i = 1 : numel(nonlocal_lange_bowsher_mr_voxels.images)
    imshow(frames(:,:,i)), colormap(gca,hot), hold on,
    red = cat(3, 1*ones(size(mrSlices)), 1*ones(size(mrSlices)), 1*ones(size(mrSlices)));
    h = imshow(red); hold off;
    set(h, 'AlphaData', mrSlices);
    fused_frame(:,:,:,i) = get(h, 'CData');
    %colormap(:,:,i) = get(gca,'Colormap');
end

filename = [outputPath sprintf('fused_slice%d_hot',slice)];
writeAnimatedGif(fused_frame, filename, 0.1,colormap(:,:,i), 0.5);
text((c-1)*sizeFrame(2)+10, (r-1)*sizeFrame(1)+10, num2str(i), 'FontSize', 14, 'FontWeight', 'bold', 'color', 'k');