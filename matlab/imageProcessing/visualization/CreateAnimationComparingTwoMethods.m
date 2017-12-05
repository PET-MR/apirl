%  *********************************************************************
%  Visualization tools.  
%  Author: Martín Belzunce. Kings College London.
%  Date: 05/12/2017
%  *********************************************************************

% Creates an animated gif with the comparison between two reconstructed
% 3D images. The images are received in a cell array, where each element of
% the cell array is a 3d matrix for a given parameter (for example, update
% number or regularization parameter).
% The animation can consist of a sequence of different way of comparing the
% reconstructed images. The order of the sequence can be chosen using the
% following values:
% 0: shows the same slice evolving as function of an input parameter. In
% this case the slice is fixed and the index of the cell array is used to
% go through the images.
% 1: shows a set of slices for a given image of the cell array.
% 2: shows rotating MIPs for a given image of the cell array.

% Parameters:
% - imagesMethod1: cell array with 3d images of method 1
% - imagesMethod2: cell array with 3d images of method 2
% - sequence: vector with the sequence order, for example [0 1 2]
% - colormap: colormap to be sued in the gif
% - outputFilename: filename of the animated gif.
% - params: at the end of the above parameters, a aprameter structure is
% expected for each element of the sequence. There are a set of common
% parameters and others are dependen on the sequence:

% Sequence 0:
% opts0.slice = 80;
% opts0.rows1 = 200:480;
% opts0.cols1 = 200:480;
% opts0.rows2 = 1 : size(nonlocal_lange_bowsher_mr_voxels.images{1},1);
% opts0.cols2 = 1 : size(nonlocal_lange_bowsher_mr_voxels.images{1},2);
% opts0.cellArrayElements = [1:19 20:2:28 30:4:70 80:10:150 160:20:300]; % Iterations I want to show
% opts0.title = '                                            Reconstruction';
% opts0.labelMethod1 = '                   PET Only';
% opts0.labelMethod2 = '              MR-Assisted PET';
% opts0.fontName = 'arial';
%
% Sequence 1:
% opts1.slices = 25:100;
% opts1.rows1 = 200:480;
% opts1.cols1 = 200:480;
% opts1.rows2 = 1 : size(nonlocal_lange_bowsher_mr_voxels.images{1},1);
% opts1.cols2 = 1 : size(nonlocal_lange_bowsher_mr_voxels.images{1},2);
% opts1.cellArrayElement = 300; % Iteration I want to show
% opts1.title = '                                            Final Images';
% opts1.labelMethod1 = '                   PET Only';
% opts1.labelMethod2 = '              MR-Assisted PET';
% opts1.fontName = 'arial';

% Sequence 2:
% opts2.angles = 0:3:357;
% opts2.rows1 = 200:480;
% opts2.cols1 = 200:480;
% opts2.rows2 = 1 : size(nonlocal_lange_bowsher_mr_voxels.images{1},1);
% opts2.cols2 = 1 : size(nonlocal_lange_bowsher_mr_voxels.images{1},2);
% opts2.cellArrayElement = 300; % Iteration I want to show
% opts2.title = '                                            Final Images';
% opts2.labelMethod1 = '                   PET Only';
% opts2.labelMethod2 = '              MR-Assisted PET';
% opts2.fontName = 'arial';
% recommendation: use even number of rows and columns
function [frames, outputFilename] = CreateAnimationComparingTwoMethods(imagesMethod1, imagesMethod2, sequence, colormap, outputFilename, varargin)

fixedArgs = 5;
if nargin ~= (fixedArgs + numel(sequence))
    error('A params structure needs to be define for each element of the sequence. Example: CreateAnimationComparingTwoMethods(imagesMethod1, imagesMethod2, [1 2], params1, params2)');
end
current_frame = 0;
frames = [];
for i = 1 : numel(sequence)
    if sequence(i) == 0
        % Show an specific slice changing.
        % Load parameters       
        sliceToShow = varargin{i}.slice;
        rowIndices1 = varargin{i}.rows1;
        colIndices1 = varargin{i}.cols1;
        rowIndices2 = varargin{i}.rows2;
        colIndices2 = varargin{i}.cols2;
        image_indices = varargin{i}.cellArrayElements;
        title = varargin{i}.title;
        label1 = varargin{i}.labelMethod1;
        label2 = varargin{i}.labelMethod2;
        fontName = varargin{i}.fontName;
        for j = 1 : numel(image_indices)
            frames(:,:,current_frame+j) = [imagesMethod1{image_indices(j)}(rowIndices1,colIndices1,sliceToShow) imagesMethod2{image_indices(j)}(rowIndices2,colIndices2,sliceToShow)];
            % Insert text:
            aux = insertText(frames(:,:,current_frame+j), [1 size(frames,1)-35; round(size(frames,2)/2) size(frames,1)-35; 1 1], {label1, label2, title}, 'Font', fontName, 'BoxOpacity', 0.0, 'FontSize', 18, 'TextColor', 'white', 'BoxColor', 'yellow');%, 'Font', 'Arial');
            frames(:,:,current_frame+j) = rgb2gray(aux);
        end
        current_frame = current_frame + numel(image_indices);

    elseif sequence(i) == 1
        % Go through the slices of an image
        slicesToShow = varargin{i}.slices;
        rowIndices1 = varargin{i}.rows1;
        colIndices1 = varargin{i}.cols1;
        rowIndices2 = varargin{i}.rows2;
        colIndices2 = varargin{i}.cols2;
        imageIndex = varargin{i}.cellArrayElement;
        title = varargin{i}.title;
        label1 = varargin{i}.labelMethod1;
        label2 = varargin{i}.labelMethod2;
        fontName = varargin{i}.fontName;
       
        for j = 1 : numel(slicesToShow)% 
            frames(:,:,current_frame+j) = [imagesMethod1{imageIndex}(rowIndices1,colIndices1,slicesToShow(j)) imagesMethod2{imageIndex}(rowIndices2,colIndices2,slicesToShow(j))];
            aux = insertText(frames(:,:,current_frame+j), [1 size(frames,1)-35; round(size(frames,2)/2) size(frames,1)-35; 1 1], {label1, label2, title}, 'Font', fontName, 'BoxOpacity', 0.0, 'FontSize', 18, 'TextColor', 'white', 'BoxColor', 'yellow');%, 'Font', 'Arial');
            frames(:,:,current_frame+j) = rgb2gray(aux);
        end
        current_frame = current_frame + numel(slicesToShow);
    elseif sequence(i) == 2
        % Rotating mips
        % Go through the slices of an image
        rowIndices1 = varargin{i}.rows1;
        colIndices1 = varargin{i}.cols1;
        rowIndices2 = varargin{i}.rows2;
        colIndices2 = varargin{i}.cols2;
        imageIndex = varargin{i}.cellArrayElement;
        title = varargin{i}.title;
        label1 = varargin{i}.labelMethod1;
        label2 = varargin{i}.labelMethod2;
        fontName = varargin{i}.fontName;
        angles = varargin{i}.angles;
        for j = 1 : numel(angles)
            [mipTransverse1, mipCoronal1, mipSagital1] = getMIPs(imagesMethod1{imageIndex}(rowIndices1,colIndices1,:), angles(j));
            [mipTransverse2, mipCoronal2, mipSagital2] = getMIPs(imagesMethod2{imageIndex}(rowIndices2,colIndices2,:), angles(j));
            % Check if needs zero padding:
            if ~isempty(frames)
                if size(frames,2) ~= (size(mipCoronal1,2)+size(mipCoronal2,2))
                    additional_cols = round(abs(size(frames,2) - (size(mipCoronal1,2)+size(mipCoronal2,2)))/2);
                        
                    if size(frames,2) > (size(mipCoronal1,2)+size(mipCoronal2,2))
                        mipCoronal1 = padarray(mipCoronal1, [0 additional_cols],'both');
                        mipCoronal2 = padarray(mipCoronal2, [0 additional_cols],'both');
                    else
                        % Pad the frames
                        frames = padarray(frames, [0 additional_cols],'both');
                    end
                end
                if size(frames,1) ~= size(mipCoronal1,1)
                    additional_rows = round(abs(size(frames,1) - size(mipCoronal1,1))/2);               
                    if size(frames,1) > size(mipCoronal1,1)
                        mipCoronal1 = padarray(mipCoronal1, [additional_rows 0],'both');
                        mipCoronal2 = padarray(mipCoronal2, [additional_rows 0],'both');
                    else
                        frames = padarray(frames, [additional_rows 0],'both');
                    end
                end
                
            end
       
            frames(:,:,current_frame+j) = [mipCoronal1 mipCoronal2]; % Crop to match the size of the pther images
            aux = insertText(frames(:,:,current_frame+j), [1 size(frames,1)-35; round(size(frames,2)/2) size(frames,1)-35; 1 1], {label1, label2, title}, 'Font', fontName, 'BoxOpacity', 0.0, 'FontSize', 18, 'TextColor', 'white', 'BoxColor', 'yellow');%, 'Font', 'Arial');
            frames(:,:,current_frame+j) = rgb2gray(aux);
        end
        current_frame = current_frame + numel(angles);
    end
end
scale = 1;
writeAnimatedGif(frames, outputFilename, 0.1,colormap, scale);
end

function [mipTransverse, mipCoronal, mipSagital] = getMIPs(volume, angle)

    volume = imrotate3(volume, angle, [0 0 1], 'crop');
    % Get the Maximum Intensity ProjectionsL
    % Transverse XY:
    mipTransverse = max(volume,[],3);
    % Coronal XZ:
    mipCoronal = max(volume,[],2);
    mipCoronal = permute(mipCoronal, [3 1 2]);
    % Sagital YZ:
    mipSagital = max(volume,[],1);
    mipSagital = permute(mipSagital, [2 3 1]);
end