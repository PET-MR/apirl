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
% - params:
% recommendation: use even number of rows and columns
function [frames, outputFilename] = CreateAnimationComparingTwoMethods(imagesMethod1, imagesMethod2, sequence, colormap, outputFilename, varargin)

fixedArgs = 5;
if nargin ~= (fixedArgs + numel(sequence))
    error('A params structure needs to be define for each element of the sequence. Example: CreateAnimationComparingTwoMethods(imagesMethod1, imagesMethod2, [1 2], params1, params2)');
end
current_frame = 0;
frames = [];
for i = 1 : numel(sequence)
    if sequence == 0
        % Show an specific slice changing.
        % Load parameters       
        sliceToShow = varargin{fixedArgs+i}.slice;
        rowIndices = varargin{fixedArgs+i}.rows;
        colIndices = varargin{fixedArgs+i}.cols;
        image_indices = varargin{fixedArgs+i}.cellArrayElements;
        title = varargin{fixedArgs+i}.title;
        label1 = varargin{fixedArgs+i}.labelMethod1;
        label2 = varargin{fixedArgs+i}.labelMethod2;
        fontName = varargin{fixedArgs+i}.fontName;
        for j = 1 : numel(image_indices)
            frames(:,:,current_frame+j) = [imagesMethod1{image_indices(j)}(rowIndices,colIndices,sliceToShow) imagesMethod2{image_indices(j)}(rowIndices,colIndices,sliceToShow)];
            % Insert text:
            aux = insertText(frames(:,:,current_frame+j), [60 numel(rowIndices)-35], label1, 'Font', fontName, 'BoxOpacity', 0.0, 'FontSize', 18, 'TextColor', 'white', 'BoxColor', 'yellow');%, 'Font', 'Arial');
            aux = rgb2gray(aux(:,:,:,i));
            aux = insertText(aux, [60+numel(colIndices) numel(rowIndices)-35], label2, 'Font', fontName, 'BoxOpacity', 0.0, 'FontSize', 18, 'TextColor', 'white', 'BoxColor', 'yellow');%, 'Font', 'Arial');
            aux = rgb2gray(aux);
            aux = insertText(aux, [numel(colIndices)-75 1], title, 'Font', fontName, 'BoxOpacity', 0.0, 'FontSize', 18, 'TextColor', 'white', 'BoxColor', 'yellow');%, 'Font', 'Arial');
            frames(:,:,i) = rgb2gray(aux);
        end
        current_frame = current_frame + numel(image_indices);

    elseif sequence == 1
        % Go through the slices of an image
        slicesToShow = varargin{fixedArgs+i}.slices;
        rowIndices = varargin{fixedArgs+i}.rows;
        colIndices = varargin{fixedArgs+i}.cols;
        imageIndex = varargin{fixedArgs+i}.cellArrayElement;
        title = varargin{fixedArgs+i}.title;
        label1 = varargin{fixedArgs+i}.labelMethod1;
        label2 = varargin{fixedArgs+i}.labelMethod2;
        fontName = varargin{fixedArgs+i}.fontName;
       
        for j = 1 : numel(slicesToShow)% 
            frames(:,:,current_frame+j) = [imagesMethod1{imageIndex}(rowIndices,colIndices,slicesToShow(j)) imagesMethod2{imageIndex}(rowIndices,colIndices,slicesToShow(j))];
            aux = insertText(frames(:,:,current_frame+j), [60 numel(rowIndices)-35], label1, 'Font', fontName, 'BoxOpacity', 0.0, 'FontSize', 18, 'TextColor', 'white', 'BoxColor', 'yellow');%, 'Font', 'Arial');
            aux = rgb2gray(aux);
            aux = insertText(aux, [60+numel(colIndices) numel(rowIndices)-35], label2, 'Font', fontName, 'BoxOpacity', 0.0, 'FontSize', 18, 'TextColor', 'white', 'BoxColor', 'yellow');%, 'Font', 'Arial');
            aux = rgb2gray(aux);
            aux = insertText(aux, [numel(colIndices)-60 1], title, 'Font', fontName, 'BoxOpacity', 0.0, 'FontSize', 18, 'TextColor', 'white', 'BoxColor', 'yellow');%, 'Font', 'Arial');
            frames(:,:,current_frame+j) = rgb2gray(aux);
        end
        current_frame = current_frame + numel(image_indices);
    elseif sequence == 2
        % Rotating mips
        % Go through the slices of an image
        rowIndices = varargin{fixedArgs+i}.rows;
        colIndices = varargin{fixedArgs+i}.cols;
        imageIndex = varargin{fixedArgs+i}.cellArrayElement;
        title = varargin{fixedArgs+i}.title;
        label1 = varargin{fixedArgs+i}.labelMethod1;
        label2 = varargin{fixedArgs+i}.labelMethod2;
        fontName = varargin{fixedArgs+i}.fontName;
        angles = varargin{fixedArgs+i}.angles;
        for j = 1 : numel(angles)
            [mipTransverse1, mipCoronal1, mipSagital1] = getMIPs(imagesMethod1{imageIndex}(rowIndices,colIndices,:), angles(i));
            [mipTransverse2, mipCoronal2, mipSagital2] = getMIPs(imagesMethod2{imageIndex}, angles(i));
            % Check if needs zero padding:
            if ~isempty(frames)
                if size(frames,2) ~= size(mipCoronal,2)
                    additional_cols = round(abs(size(frames,2) - size(mipCoronal,2))/2);
                        
                    if size(frames,2) > size(mipCoronal,2)
                        mipCoronal1_filled = padarray(mipCoronal1, [0 additional_cols],'both');
                        mipCoronal2_filled = padarray(mipCoronal2, [0 additional_cols],'both');
                    else
                        % Pad the frames
                        frames = padarray(frames, [0 additional_cols],'both');
                    end
                end
                if size(frames,1) ~= size(mipCoronal,1)
                    additional_rows = round(abs(size(frames,1) - size(mipCoronal,1))/2);               
                    if size(frames,1) > size(mipCoronal,1)
                        mipCoronal1_filled = padarray(mipCoronal1, [additional_rows 0],'both');
                        mipCoronal2_filled = padarray(mipCoronal2, [additional_rows 0],'both');
                    else
                        frames = padarray(frames, [additional_rows 0],'both');
                    end
                end
                
            end
       
            frames(:,:,current_frame+j) = [mipCoronal1_filled mipCoronal2_filled]; % Crop to match the size of the pther images
            aux = insertText(frames(:,:,current_frame+j), [60 size(frames,1)-35], label1, 'Font', fontName, 'BoxOpacity', 0.0, 'FontSize', 18, 'TextColor', 'white', 'BoxColor', 'yellow');%, 'Font', 'Arial');
            aux = rgb2gray(aux);
            aux = insertText(aux, [60+numel(colIndices) size(frames,1)-35], label2, 'Font', fontName, 'BoxOpacity', 0.0, 'FontSize', 18, 'TextColor', 'white', 'BoxColor', 'yellow');%, 'Font', 'Arial');
            aux = rgb2gray(aux);
            aux = insertText(aux, [numel(colIndices)-75 1], title, 'Font', fontName, 'BoxOpacity', 0.0, 'FontSize', 18, 'TextColor', 'white', 'BoxColor', 'yellow');%, 'Font', 'Arial');
            frames(:,:,current_frame+j) = rgb2gray(aux);
        end
        current_frame = current_frame + numel(image_indices);
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