%  *********************************************************************
%  Visualization tools.  
%  Author: Martín Belzunce. Kings College London.
%  Date: 05/12/2017
%  *********************************************************************

% Examples of how to use the function CreateAnimationComparingTwoMethods.

% Firs for individual type of animation and then for a sequence.
%% GENERATE ANIMATION WITH FUNCTION
% First go only through iterations
opts0.slice = 80;
opts0.rows1 = 200:480;
opts0.cols1 = 200:480;
opts0.rows2 = 1 : size(nonlocal_lange_bowsher_mr_voxels.images{1},1);
opts0.cols2 = 1 : size(nonlocal_lange_bowsher_mr_voxels.images{1},2);
opts0.cellArrayElements = [1:19 20:2:28 30:4:70 80:10:150 160:20:300]; % Iterations I want to show
opts0.title = '                                            Reconstruction';
opts0.labelMethod1 = '                   PET Only';
opts0.labelMethod2 = '              MR-Assisted PET';
opts0.fontName = 'arial';
[frames, outputFilename] = CreateAnimationComparingTwoMethods(mlem_resampled, nonlocal_lange_bowsher_mr_voxels.images, [0], gray, sprintf('/data/Results/FDG_11/OnlyIterationsSlice_%d.gif', opts0.slice), opts0);

%% Only through slices
opts1.slices = 25:100;
opts1.rows1 = 200:480;
opts1.cols1 = 200:480;
opts1.rows2 = 1 : size(nonlocal_lange_bowsher_mr_voxels.images{1},1);
opts1.cols2 = 1 : size(nonlocal_lange_bowsher_mr_voxels.images{1},2);
opts1.cellArrayElement = 300; % Iteration I want to show
opts1.title = '                                            Final Images';
opts1.labelMethod1 = '                   PET Only';
opts1.labelMethod2 = '              MR-Assisted PET';
opts1.fontName = 'arial';
[frames, outputFilename] = CreateAnimationComparingTwoMethods(mlem_resampled, nonlocal_lange_bowsher_mr_voxels.images, [1], gray, sprintf('/data/Results/FDG_11/OnlySlicesIteration_%d.gif', opts1.cellArrayElement), opts1);

%% Only through MIPs
opts2.angles = 0:15:345;% 0:3:357;
opts2.rows1 = 200:480;
opts2.cols1 = 200:480;
opts2.rows2 = 1 : size(nonlocal_lange_bowsher_mr_voxels.images{1},1);
opts2.cols2 = 1 : size(nonlocal_lange_bowsher_mr_voxels.images{1},2);
opts2.cellArrayElement = 300; % Iteration I want to show
opts2.title = '                                            Final Images';
opts2.labelMethod1 = '                   PET Only';
opts2.labelMethod2 = '              MR-Assisted PET';
opts2.fontName = 'arial';
[frames, outputFilename] = CreateAnimationComparingTwoMethods(mlem_resampled, nonlocal_lange_bowsher_mr_voxels.images, [2], gray, '/data/Results/FDG_11/RotatingMips.gif', opts2);
%% Create a Sequence
opts1.slices = [opts0.slice:100 25:opts0.slice-1]; % Change the slices to start from the same slice as the iteration update
[frames, outputFilename] = CreateAnimationComparingTwoMethods(mlem_resampled, nonlocal_lange_bowsher_mr_voxels.images, [0 1 2], gray, '/data/Results/FDG_11/FullSequence.gif', opts0, opts1, opts2);