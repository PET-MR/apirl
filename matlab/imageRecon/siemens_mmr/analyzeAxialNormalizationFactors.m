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
addpath('/workspaces/Martin/KCL/Biograph_mMr/mmr');
apirlPath = '/workspaces/Martin/PET/apirl-code/trunk/';
addpath(genpath([apirlPath '/matlab']));
setenv('PATH', [getenv('PATH') ':/workspaces/Martin/PET/apirl-code/trunk/build/bin']);
setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') ':/workspaces/Martin/PET/apirl-code/trunk/build//bin']);
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

%% PLOT AXIAL COMPONENTS
% Main axial effects:
figure;
hold on;
for i = 1 : numel(normFiles)
    plot(componentFactors{i}{4});
end;
title(sprintf('Main Axial Factors for %d Normalization Files', numel(normFiles)));
ylabel('Factor');
xlabel('Number of Sinogram');
set(gcf, 'position', [1 25 1920 1069]);
%% PLOT OTHER AXIAL COMPONENTS
% Other axial effects:
figure;
hold on;
for i = 1 : numel(normFiles)
    plot(componentFactors{i}{8});
end;
ylim([0.9 1.1]);
title(sprintf('Other Axial Factors for %d Normalization Files', numel(normFiles)));
ylabel('Factor');
xlabel('Number of Sinogram');
set(gcf, 'position', [1 25 1920 1069]);

% Find the bad one:
numFiles = numel(normFiles);
i = 1;
while i <= numFiles
    cantOdds = find(componentFactors{i}{8}>2);
    if(~isempty(cantOdds))
        figure;
        plot(componentFactors{i}{8});
        title(sprintf('Bad Other Axial Factors: %s', normFiles(i).name));
        ylabel('Factor');
        xlabel('Number of Sinogram');
        set(gcf, 'position', [1 25 1920 1069]);
        % Take it out from de list:
        normFiles(i) = [];
        componentFactors(i) = [];
        i = i-1;
        numFiles = numFiles - 1;
    end
    i = i + 1;
end;

%% STATS OF THE AXIAL COMPONENTS
factorFromSpan = 1./structSizeSino3dSpan11.numSinosMashed;
% Normalize to the mean:
factorFromSpan = factorFromSpan ./ mean(factorFromSpan);
% Axial components
allComponents = zeros(numel(normFiles), numel(componentFactors{1}{4}));
allOtherComponents = zeros(numel(normFiles), numel(componentFactors{1}{8}));
for i = 1 : numel(normFiles)
    allComponents(i,:) = [componentFactors{i}{4}]';
    allOtherComponents(i,:) = [componentFactors{i}{8}]';
end;
indexSinos = 1 : numel(allComponents(1,:));
meanMain = mean(allComponents);
devMain = std(allComponents);
meanOther = mean(allOtherComponents);
devOther = std(allOtherComponents);
%% PLOT SOME STATS FOR PUBLICATION IN TECHNICAL NOTE
graphsPath = '/home/mab15/workspace/KCL/Publications/svn/impact_crystal_efficiencies_2015/';
% plot:
figure;
set(gcf, 'Position', [50 50 1600 1000]);
subplot(2,1,1);
plot(indexSinos, meanMain, indexSinos, meanOther, 'LineWidth', 2);
legend('Mean Main Axial Factors', 'Mean Other Axial Factors', 'Location','NorthWest');
title('a) Mean Value of Axial Factors');
ylabel('Factor', 'FontSize', 16, 'FontWeight', 'bold');
xlabel('i_{z}', 'FontSize', 16, 'FontWeight', 'bold');
ticklabels = get(gca, 'XtickLabel');
set(gca, 'XtickLabel', ticklabels, 'FontSize',16);
ticklabels = get(gca, 'YtickLabel');
set(gca, 'YtickLabel', ticklabels, 'FontSize',16);


% plot:
subplot(2,1,2);
plot(indexSinos, devMain./meanMain, indexSinos, devOther./meanOther, 'LineWidth', 2);
legend('StdDev/Mean Main Axial Factors', 'StdDev/Mean Other Axial Factors', 'Location','NorthWest');
title('b) Normalized StdDev of Axial Factors');
ylabel('Factor', 'FontSize', 16, 'FontWeight', 'bold');
xlabel('i_{z}', 'FontSize', 16, 'FontWeight', 'bold');
ticklabels = get(gca, 'XtickLabel');
set(gca, 'XtickLabel', ticklabels, 'FontSize',16);
ticklabels = get(gca, 'YtickLabel');
set(gca, 'YtickLabel', ticklabels, 'FontSize',16);
% Save for publication:
set(gcf,'PaperPositionMode','auto');    % Para que lo guarde en el tamaño modificado.
fullFilename = [graphsPath 'figure3a_3b'];
saveas(gca, [fullFilename], 'tif');
saveas(gca, [fullFilename], 'epsc');
%% COMPARE WITH NUM SINOS
factorFromSpan = 1./structSizeSino3dSpan11.numSinosMashed;
% Normalize to the mean:
factorFromSpan = factorFromSpan ./ mean(factorFromSpan);
% Plot and compare with one normalization file:
figure;
plot([componentFactors{1}{4} factorFromSpan'], 'LineWidth', 2);
title('Comparison of Influence of Span and the Axial Factor'); 
legend('Axial Factor from CBN', 'Span 11 Factors');
ylabel('Factor');
xlabel('Number of Sinogram');
set(gcf, 'position', [1 25 1920 1069]);