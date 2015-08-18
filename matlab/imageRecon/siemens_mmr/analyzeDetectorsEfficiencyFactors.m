%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 16/02/2015
%  *********************************************************************
%  This scripts analyze the axial factors. It comapres the one in the .n
%  file
clear all 
close all
%% PATHS FOR EXTERNAL FUNCTIONS AND RESULTS
addpath('/home/mab15/workspace/Biograph_mMr/mmr');
apirlPath = '/home/mab15/workspace/apirl-code/trunk/';
addpath(genpath([apirlPath '/matlab']));
setenv('PATH', [getenv('PATH') ':/home/mab15/workspace/apirl-code/trunk/build/bin']);
setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') ':/home/mab15/workspace/apirl-code/trunk/build/bin']);
normPath = '/home/mab15/workspace/KCL/Biograph_mMr/Normalization/NormFiles/';
%setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') ':/usr/lib/x86_64-linux-gnu/']);
%% GET ALL THE NORM FILES IN THE PATH AND READ THEM
normFiles = dir([normPath '*.n']);
for i = 1 : numel(normFiles)
    componentFactors{i}  = readmMrComponentBasedNormalization([normPath normFiles(i).name], 0);
end
% Get the date of each date:
for i = 1 : numel(normFiles)
    datenumber(i) = datenum(normFiles(i).date);
end
% Sort it:
[datenumber_ordered, indexes] = sort(datenumber);
daysFromFirst = datenumber_ordered - datenumber_ordered(1);
for i = 1 : numel(normFiles)
    componentFactorsOrdered{i} =  componentFactors{indexes(i)};
end
%% GENERATE THE STRUCT OF A SINO3D SPAN 11
% In that structure I have the amount of sinograms axially compressed per
% stored sinogram.
numR = 344; numTheta = 252; numZ = 64; maxAbsRingDiff = 60; rFov_mm = 594/2; zFov_mm = 258;
structSizeSino3dSpan11 = getSizeSino3dFromSpan(numR, numTheta, numZ, ...
    rFov_mm, zFov_mm, 11, maxAbsRingDiff);
numCrystalsPerRing = size(componentFactorsOrdered{1}{3},1);
numRings = size(componentFactorsOrdered{1}{3},2);
%% GET CRYSTAL EFFICIENCIES
% Get the crystal efficiencies:
crystalEff = zeros(size(componentFactorsOrdered{1}{3},1), size(componentFactorsOrdered{1}{3},2), numel(componentFactorsOrdered));   % A 3d maitrx: crystal element, ring, norm file
for i = 1 : numel(componentFactorsOrdered)
    crystalEff(:,:,i) = componentFactorsOrdered{i}{3};
end;

%% STATS OF THE CRYSTAL EFFICIENCIES
% Global variations in a normalization file:
for i = 1 : numel(componentFactorsOrdered)
    devGlobal(i) = std2(crystalEff(:,:,i));
    meanGlobal(i) = mean2(crystalEff(:,:,i));
end
figure;
plot(devGlobal);
title('Standard Deviation for Each Normalization File');
ylabel('Std Dev of Efficiency Factors');
xlabel('Norm File');

% % Get std dev for each pixel position:
% devCrystalEff = std(crystalEff,0,3);
% meanCrystalEff = mean(crystalEff,3);
% % Plot for each rings:
% for i = 1 : 9 : numRings
%     figure;
%     h=errorbar(meanCrystalEff(:,i),devCrystalEff(:,i));
%     title(sprintf('Standard Deviation of Crystal Efficiencies in Ring %d during 1 1/2 year', i));
%     xlim([0 numCrystalsPerRing]);
%     ylabel('Efficiency Factor');
%     xlabel('Crystal Element');
%     set(gcf, 'position', [1 25 1920 1069]);
% end
% %% GET SOME EXAMPLES
% % Maximum deviation in ring:
% [maxDevValue, maxDevPos] = max(devCrystalEff);
% figure;
% plot(permute(crystalEff(maxDevPos(1),1,:), [3 1 2]));
% title(sprintf('MAX EXAMPLE: Variation of Crystal %d in ring 1 in 1 1/2 year', maxDevPos(1)));
% ylabel('Efficiency Factor');
% xlabel('Crystal Element');
% set(gcf, 'position', [1 25 1920 1069]);
% 
% % Minimum deviation in ring:
% [minDevValue, minDevPos] = min(devCrystalEff);
% figure;
% plot(permute(crystalEff(minDevPos(1),1,:), [3 1 2]));
% title(sprintf('MIN EXAMPLE: Variation of Crystal %d in ring 1 in 1 1/2 year', minDevPos(1)));

%% REMOVE NORMALIZATION FILES WITH BAD VALUES
% Put a threshold to consider a failed norm file or failed detector:
threshold = 0.5;
[c, r, n] = ind2sub([size(crystalEff,1), size(crystalEff,2), size(crystalEff,3)],find((crystalEff < (1-threshold)) | (crystalEff > (1+threshold))));
crystalEffFiltered = crystalEff;
crystalEffFiltered(:,:,n) = [];
daysFromFirst(n) = [];
% Recompute standard deviations:
% Global variations in a normalization file:
devGlobal = zeros(1, size(crystalEffFiltered,3));
meanGlobal = zeros(1, size(crystalEffFiltered,3));
for i = 1 : size(crystalEffFiltered,3)
    aux = crystalEffFiltered(:,:,i);
    vector = aux(:);
    devGlobal(i) = std(vector);
    meanGlobal(i) = mean(vector);
    medianGlobal(i) = median(vector);
    minGlobal(i) = min(min(crystalEffFiltered(:,:,i)));
    maxGlobal(i) = max(max(crystalEffFiltered(:,:,i)));
    
end
mean(devGlobal)
%% PLOT SOME STATS FOR PUBLICATION IN TECHNICAL NOTE
graphsPath = '/home/mab15/workspace/KCL/Publications/svn/impact_crystal_efficiencies_2015/';
figure;
errorbar(daysFromFirst, meanGlobal,devGlobal, '-o','LineWidth',2);
hold on
set(gcf, 'Position', [50 50 1600 1000]);
plot(daysFromFirst, medianGlobal, '-x', daysFromFirst, minGlobal, '-s', daysFromFirst, maxGlobal, '-d', ...
    daysFromFirst, permute(crystalEffFiltered(200,48,:), [3 1 2]), '-^','LineWidth', 2, 'MarkerSize',12);
xlim([min(daysFromFirst) max(daysFromFirst)]);
legend('Mean \pm StdDev', 'Median', 'Maximum', 'Minimum', 'Crystal 200 in Ring 48', 'Location','NorthWest');
title('Crystal Efficiencies on Time', 'FontSize', 16, 'FontWeight', 'bold');
xlabel('Days', 'FontSize', 16, 'FontWeight', 'bold');
ylabel('Crystal Efficiencies', 'FontSize', 16, 'FontWeight', 'bold');
ticklabels = get(gca, 'XtickLabel');
set(gca, 'XtickLabel', ticklabels, 'FontSize',16);
ticklabels = get(gca, 'YtickLabel');
set(gca, 'YtickLabel', ticklabels, 'FontSize',16);
% Save for publication:
set(gcf,'PaperPositionMode','auto');    % Para que lo guarde en el tamaño modificado.
fullFilename = [graphsPath 'figure2'];
saveas(gca, [fullFilename], 'tif');
saveas(gca, [fullFilename], 'epsc');
%plotyy(daysFromFirst, meanGlobal,daysFromFirst, devGlobal)
%%
figure;
plot(daysFromFirst, permute(crystalEffFiltered(200,48,:), [3 1 2]), '-s', daysFromFirst, permute(crystalEffFiltered(300,32,:), [3 1 2]), '-d', ...
    daysFromFirst, permute(crystalEffFiltered(100,15,:), [3 1 2]), '-^', 'LineWidth', 2, 'MarkerSize',12);
title('Crystal Efficiency');
ylabel('Crystal Efficiency');
xlabel('Norm File');
set(gcf, 'position', [1 25 1920 1069]);
legend('Crystal 200 in Ring 48', 'Crystal 300 in Ring 32', 'Crystal 100 in Ring 15', 'Location','NorthWest');
title('Crystal Efficiencies on Time', 'FontSize', 16, 'FontWeight', 'bold');
xlabel('Days', 'FontSize', 16, 'FontWeight', 'bold');
ylabel('Crystal Efficiencies', 'FontSize', 16, 'FontWeight', 'bold');
ticklabels = get(gca, 'XtickLabel');
set(gca, 'XtickLabel', ticklabels, 'FontSize',16);
ticklabels = get(gca, 'YtickLabel');
set(gca, 'YtickLabel', ticklabels, 'FontSize',16);
% Save for publication:
set(gcf,'PaperPositionMode','auto');    % Para que lo guarde en el tamaño modificado.
fullFilename = [graphsPath 'figure2a'];
saveas(gca, [fullFilename], 'tif');
saveas(gca, [fullFilename], 'epsc');
%% HISTOGRAMS
[minDev, minIndex] = min(devGlobal);
[maxDev, maxIndex] = max(devGlobal);
medianDev = median(devGlobal);
indMedian = find(devGlobal>(medianDev*0.999) & devGlobal<(medianDev*1.0001));
minCrystals = crystalEffFiltered(:,:,minIndex);
maxCrystals = crystalEffFiltered(:,:,maxIndex);
medianCrystals = crystalEffFiltered(:,:,indMedian); 
[countsMin, valoresMin] = hist(minCrystals(:),0.8:0.01:1.20);
[countsMax, valoresMax] = hist(maxCrystals(:),0.8:0.01:1.20);
[countsMedian, valoresMedian] = hist(medianCrystals(:),0.8:0.01:1.20);
figure;
plot(valoresMin, countsMin, valoresMax, countsMax, valoresMedian, countsMedian, 'LineWidth', 3, 'MarkerSize',12);
ylabel('Counts', 'FontSize', 16, 'FontWeight', 'bold');
xlabel('Crystal Efficiencies', 'FontSize', 16, 'FontWeight', 'bold');
title('Crystal Efficiencies Distributions', 'FontSize', 16, 'FontWeight', 'bold');
set(gcf, 'Position', [50 50 1600 1000]);
legend('Minimum StdDev', 'Maximum StdDev', 'Median StdDev', 'Location','NorthWest');
ticklabels = get(gca, 'XtickLabel');
set(gca, 'XtickLabel', ticklabels, 'FontSize',16);
ticklabels = get(gca, 'YtickLabel');
set(gca, 'YtickLabel', ticklabels, 'FontSize',16);
% Save for publication:
set(gcf,'PaperPositionMode','auto');    % Para que lo guarde en el tamaño modificado.
fullFilename = [graphsPath 'figure3'];
saveas(gca, [fullFilename], 'tif');
saveas(gca, [fullFilename], 'epsc');