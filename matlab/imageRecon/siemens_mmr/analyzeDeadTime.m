%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 16/02/2015
%  *********************************************************************
%  This scripts analyze the dead time factors. It comapres the one in the .n
%  file
clear all 
close all
%% PATHS FOR EXTERNAL FUNCTIONS AND RESULTS
addpath('/home/mab15/workspace/KCL/Biograph_mMr/mmr');
apirlPath = '/home/mab15/workspace/apirl-code/trunk/';
addpath(genpath([apirlPath '/matlab']));
setenv('PATH', [getenv('PATH') ':/home/mab15/workspace/apirl-code/trunk/build/bin']);
setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') ':/home/mab15/workspace/apirl-code/trunk/build//bin']);
normPath = '/home/mab15/workspace/KCL/Biograph_mMr/Normalization/NormFiles/';
%setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') ':/usr/lib/x86_64-linux-gnu/']);
%% READ A SINOGRAM
% Get two sinograms with different time to evaluate dead time:
% To get single data example:
filenameUncompressedMmr = '/home/mab15/workspace/KCL/Biograph_mMr/mmr/5hr_ge68/cylinder_5hours.s.hdr';
outFilenameIntfSinograms = '/home/mab15/workspace/KCL/Biograph_mMr/mmr/5hr_ge68/cylinder_5hoursIntf';
[structInterfile1, structSizeSino] = getInfoFromSiemensIntf(filenameUncompressedMmr);
[sinogram1, delayedSinogram1, structSizeSino3d] = getIntfSinogramsFromUncompressedMmr('/home/mab15/workspace/KCL/Biograph_mMr/mmr/5hr_ge68/cylinder_5hours.s', outFilenameIntfSinograms);

filenameUncompressedMmr = '/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/2601/2601_inf/PET_ACQ_16_20150116131121-0.s.hdr';
outFilenameIntfSinograms = '/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/2601/2601_inf/PET_ACQ_16_Intf';
[structInterfile2, structSizeSino] = getInfoFromSiemensIntf(filenameUncompressedMmr);
[sinogram2, delayedSinogram2, structSizeSino3d] = getIntfSinogramsFromUncompressedMmr('/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/2601/2601_inf/PET_ACQ_16_20150116131121-0.s', outFilenameIntfSinograms);
%% GET ALL THE NORM FILES IN THE PATH AND READ THEM
normFiles = dir([normPath '*.n']);
for i = 1 : numel(normFiles)
    [componentFactors{i} componentLabels] = readmMrComponentBasedNormalization([normPath normFiles(i).name], 0);
end
%% GENERATE THE STRUCT OF A SINO3D SPAN 11
% In that structure I have the amount of sinograms axially compressed per
% stored sinogram.
numR = 344; numTheta = 252; numZ = 64; maxAbsRingDiff = 60; rFov_mm = 594/2; zFov_mm = 258;
structSizeSino3dSpan11 = getSizeSino3dFromSpan(numR, numTheta, numZ, ...
    rFov_mm, zFov_mm, 11, maxAbsRingDiff);
%% GET PARALYZING AND NON PARALYZING DEAD TIME
% Get the crystal efficiencies:
parDeadTime = zeros(size(componentFactors{1}{5},1), numel(normFiles));   % A 2d maitrx
nonParDeadTime = zeros(size(componentFactors{1}{6},1), numel(normFiles));   % A 2d maitrx
for i = 1 : numel(normFiles)
    parDeadTime(:,i) = componentFactors{i}{5};
    nonParDeadTime(:,i) = componentFactors{i}{6};
end
%% PARALYZING DEAD TIME 
% Check if they vary with the time:
parDeadTimeRef = parDeadTime(:,1);
numDiffFactors = 0;
for i = 1 : numel(normFiles)
    diff = parDeadTime(:,1) ~= parDeadTimeRef;
    if(sum(diff)>0)
        disp(sprintf('The paralyzing dead time factors for the %s file was different from %s.', normFiles(i).name, normFiles(1).name));
        numDiffFactors = numDiffFactors +1;
    end
end
if (numDiffFactors == 0)
    disp('The paralyzing dead time Factors are the same for all the norm files.');
end

%% NON PARALYZING DEAD TIME 
% Check if they vary with the time:
nonParDeadTimeRef = nonParDeadTime(:,1);
numDiffFactors = 0;
for i = 1 : numel(normFiles)
    diff = nonParDeadTime(:,1) ~= nonParDeadTimeRef;
    if(sum(diff) > 0)
        disp(sprintf('The non paralyzing dead time factors for the %s file was different from %s.', normFiles(i).name, normFiles(1).name));
    end
end

if (numDiffFactors  == 0)
    disp('The non paralyzing dead time Factors are the same for all the norm files.');
end
%% CHECK TWO POSSIBLE FORMULAS
% 1) From Casey 1995:
singles = 0 : 1000 : 500000;
deadTimeEfficency1 = (exp(singles.*parDeadTimeRef(1)./(1+singles.*nonParDeadTimeRef(1)))./(1+singles.*nonParDeadTimeRef(1)));
% 2) From Tai 1998:
deadTimeEfficency2 =  (1 + parDeadTimeRef(1)*singles - nonParDeadTimeRef(1).*singles.*singles);
figure;
plot(singles, singles.*deadTimeEfficency1, singles, singles.*deadTimeEfficency2);
legend('Parameters As Casey 1995', 'Parameters As Tai 1998');
xlabel('Incident Singles');
ylabel('Detected Singles');
% For higher count rates:
% 1) From Casey 1995:
singles = 0 : 1000 : 1000000;
deadTimeEfficency1 = (exp(singles.*parDeadTimeRef(1)./(1+singles.*nonParDeadTimeRef(1)))./(1+singles.*nonParDeadTimeRef(1)));
% 2) From Tai 1998:
deadTimeEfficency2 =  (1 + parDeadTimeRef(1)*singles - nonParDeadTimeRef(1).*singles.*singles);
figure;
plot(singles, singles.*deadTimeEfficency1, singles, singles.*deadTimeEfficency2);
legend('Parameters As Casey 1995', 'Parameters As Tai 1998');
xlabel('Incident Singles');
ylabel('Detected Singles');
set(gcf, 'position', [1 25 1920 1069]);
%% PLOT AN ESTIMATED DEAD TIME FACTOR
% Parameters of the buckets:
numberOfTransverseBlocksPerBucket = 2;
numberOfAxialBlocksPerBucket = 1;
numberOfBuckets = structInterfile1.NumberOfBuckets;
numberofAxialBlocks = 8;
numberofTransverseBlocksPerRing = 8;
numberOfBucketsInRing = numberOfBuckets / (numberofTransverseBlocksPerRing);
numberOfBlocksInBucketRing = numberOfBuckets / (numberofTransverseBlocksPerRing*numberOfAxialBlocksPerBucket);
numberOfTransverseCrystalsPerBlock = 9; % includes the gap
numberOfAxialCrystalsPerBlock = 8;
% Get singles rate per ring:
for i = 1 : structInterfile1.NumberOfRings
    singlesRatePerRing(i) = structInterfile1.SinglesPerBucket;
end
% Correct singles rate:
correctedSinglesRate = singlesRate ./ (1 - parDeadTimeRef.*singlesRate - nonParDeadTimeRef.*singlesRate.*singlesRate);
%% PLOT COUNTS IN DIRECT SINOGRAMS
% Plot the counts the direct sinograms, correcting for decay time and dead
% time.
directSinos1 = sinogram1(1:structInterfile1.NumberOfRings);
directSinos2 = sinogram2(1:structInterfile1.NumberOfRings);
% Counts per sino:
CountsPerRing1 = sum(sum(directSinos1));
CountsPerRing1 = permute(CountsPerRing1, [3 1 2]);
CountsPerRing2 = sum(sum(directSinos2));
CountsPerRing2 = permute(CountsPerRing2, [3 1 2]);
% Correct for activity decay:
dateNum1 = datenum(structInterfile1.Date(1:4), structInterfile1.Date(6:7), structInterfile1.Date(9:10));
daysBetween21 = str2num(structInterfile1.Date())