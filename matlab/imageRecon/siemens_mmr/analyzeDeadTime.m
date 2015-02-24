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
filenameUncompressedMmr = '/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/QualityControlAcquisitions/PET_ACQ_12_20140822100449-0uncomp.s.hdr';
outFilenameIntfSinograms = '/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/QualityControlAcquisitions/PET_ACQ_16_Intf';
[structInterfile1, structSizeSino] = getInfoFromSiemensIntf(filenameUncompressedMmr);
[sinogram1, delayedSinogram1, structSizeSino3d] = getIntfSinogramsFromUncompressedMmr('/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/QualityControlAcquisitions/PET_ACQ_12_20140822100449-0uncomp.s', outFilenameIntfSinograms);


filenameUncompressedMmr = '/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/2601/UncompressedInterfile/PET_ACQ_16_20150116131121-0uncomp.s.hdr';
outFilenameIntfSinograms = '/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/2601/UncompressedInterfile/PET_ACQ_16_Intf';
[structInterfile2, structSizeSino] = getInfoFromSiemensIntf(filenameUncompressedMmr);
[sinogram2, delayedSinogram2, structSizeSino3d] = getIntfSinogramsFromUncompressedMmr('/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/2601/UncompressedInterfile/PET_ACQ_16_20150116131121-0uncomp.s', outFilenameIntfSinograms);
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
figure;
plot(singles, deadTimeEfficency1, singles, deadTimeEfficency2);
legend('Parameters As Casey 1995', 'Parameters As Tai 1998');
xlabel('Incident Singles');
ylabel('Efficiency');
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
numberOfBucketRings = structInterfile1.NumberOfRings / (numberOfAxialBlocksPerBucket*numberOfAxialCrystalsPerBlock);
% Get singles rate per ring FOR MEASUREMENT1:
for i = 1 : numberOfBucketRings
    singlesRatePerRing1((i-1)*(numberOfAxialBlocksPerBucket*numberOfAxialCrystalsPerBlock)+1:i*(numberOfAxialBlocksPerBucket*numberOfAxialCrystalsPerBlock)) = ...
        sum(structInterfile1.SinglesPerBucket((i-1)*numberOfBucketsInRing+1 : i*numberOfBucketsInRing)) ./ (numberOfAxialBlocksPerBucket*numberOfAxialCrystalsPerBlock);
end
% Get singles rate per ring FOR MEASUREMENT2:
for i = 1 : numberOfBucketRings
    singlesRatePerRing2((i-1)*(numberOfAxialBlocksPerBucket*numberOfAxialCrystalsPerBlock)+1:i*(numberOfAxialBlocksPerBucket*numberOfAxialCrystalsPerBlock)) = ...
        sum(structInterfile2.SinglesPerBucket((i-1)*numberOfBucketsInRing+1 : i*numberOfBucketsInRing)) ./ (numberOfAxialBlocksPerBucket*numberOfAxialCrystalsPerBlock);
end
% Get singles rate per block FOR MEASUREMENT1:
for i = 1 : numberOfBucketRings
    singlesRatePerBlock1((i-1)*(numberOfAxialBlocksPerBucket*numberOfAxialCrystalsPerBlock)+1:i*(numberOfAxialBlocksPerBucket*numberOfAxialCrystalsPerBlock)) = ...
        mean(structInterfile1.SinglesPerBucket((i-1)*numberOfBucketsInRing+1 : i*numberOfBucketsInRing)); % ./ (numberOfAxialBlocksPerBucket*numberOfTransverseBlocksPerBucket); % Using per bucket instead per block
end
% Get singles rate per block FOR MEASUREMENT2:
for i = 1 : numberOfBucketRings
    singlesRatePerBlock2((i-1)*(numberOfAxialBlocksPerBucket*numberOfAxialCrystalsPerBlock)+1:i*(numberOfAxialBlocksPerBucket*numberOfAxialCrystalsPerBlock)) = ...
        mean(structInterfile2.SinglesPerBucket((i-1)*numberOfBucketsInRing+1 : i*numberOfBucketsInRing)); % ./ (numberOfAxialBlocksPerBucket*numberOfTransverseBlocksPerBucket); % Using per bucket instead per block
end
%% CORRECT FOR ACTIVITY DECAY
% Correct for activity decay:
dateNum1 = datenum(str2num(structInterfile1.StudyDateYyyyMmDd(1:4)), str2num(structInterfile1.StudyDateYyyyMmDd(6:7)), str2num(structInterfile1.StudyDateYyyyMmDd(9:10)));
dateNum2 = datenum(str2num(structInterfile2.StudyDateYyyyMmDd(1:4)), str2num(structInterfile2.StudyDateYyyyMmDd(6:7)), str2num(structInterfile2.StudyDateYyyyMmDd(9:10)));
daysBetween21 = dateNum2 - dateNum1;
% Half life Ge-68:
halfLife_days = 271;
% Activity Decay Correction from measurement 2 to 1:
A2DecayFactor = exp(-log(2)/halfLife_days.*daysBetween21);
singlesRatePerRing2_DecayCorrected = singlesRatePerRing2 ./ A2DecayFactor;
singlesRatePerBlock2_DecayCorrected = singlesRatePerBlock2 ./ A2DecayFactor;
%% PLOT SINGLES RATES PER RING WITHOUT DEAD TIME CORRECTION
% Plot the singles rate with decay correction.
figure;
plot(1:structInterfile1.NumberOfRings, singlesRatePerRing1, 1:structInterfile2.NumberOfRings, singlesRatePerRing2, 1:structInterfile2.NumberOfRings, singlesRatePerRing2_DecayCorrected);
xlabel('Ring');
ylabel('Single Rates');
legend('Acquisition 1', sprintf('Acquisition 2 (%d days later)', daysBetween21), sprintf('Acquisition 2 Decay Corrected'));
set(gcf, 'position', [1 25 1920 1069]);
% %% CORRECT FOR DEAD TIME
% % Correct singles rate:
% deadtimeFactor11 = (exp(singlesRatePerRing1.*parDeadTimeRef(1)./(1+singlesRatePerRing1.*nonParDeadTimeRef(1)))./(1+singlesRatePerRing1.*nonParDeadTimeRef(1)));
% deadtimeFactor12 = (1 + parDeadTimeRef(1)*singlesRatePerRing1 - nonParDeadTimeRef(1).*singlesRatePerRing1.*singlesRatePerRing1);
% correctedSinglesRate1_DeadTimeCorrected1 = singlesRatePerRing1 ./ deadtimeFactor11;
% correctedSinglesRate1_DeadTimeCorrected2 = singlesRatePerRing1 ./ deadtimeFactor12;
% 
% % First correct dead time and then for activity, because the deadtime
% % correction must use the real single rates.
% deadtimeFactor21 = (exp(singlesRatePerRing2.*parDeadTimeRef(1)./(1+singlesRatePerRing2.*nonParDeadTimeRef(1)))./(1+singlesRatePerRing2.*nonParDeadTimeRef(1)));
% deadtimeFactor22 = (1 + parDeadTimeRef(1)*singlesRatePerRing2 - nonParDeadTimeRef(1).*singlesRatePerRing2.*singlesRatePerRing2);
% correctedSinglesRate2_DeadTimeCorrected1 = singlesRatePerRing2 ./ deadtimeFactor21;
% correctedSinglesRate2_DeadTimeCorrected2 = singlesRatePerRing2 ./ deadtimeFactor22;
% % Then correct for activity:
% correctedSinglesRate2_DeadTimeCorrected1 = correctedSinglesRate2_DeadTimeCorrected1 ./ A2DecayFactor;
% correctedSinglesRate2_DeadTimeCorrected2 = correctedSinglesRate2_DeadTimeCorrected2 ./ A2DecayFactor;
%% CORRECT FOR DEAD TIME
% Correct singles rate:
deadtimeFactor11 = (exp(singlesRatePerBlock1.*parDeadTimeRef(1)./(1+singlesRatePerBlock1.*nonParDeadTimeRef(1)))./(1+singlesRatePerBlock1.*nonParDeadTimeRef(1)));
deadtimeFactor12 = (1 + parDeadTimeRef(1)*singlesRatePerBlock1 - nonParDeadTimeRef(1).*singlesRatePerBlock1.*singlesRatePerBlock1);
correctedSinglesRate1_DeadTimeCorrected1 = singlesRatePerRing1 ./ deadtimeFactor11;
correctedSinglesRate1_DeadTimeCorrected2 = singlesRatePerRing1 ./ deadtimeFactor12;

deadtimeFactor21 = (exp(singlesRatePerBlock2.*parDeadTimeRef(1)./(1+singlesRatePerBlock2.*nonParDeadTimeRef(1)))./(1+singlesRatePerBlock2.*nonParDeadTimeRef(1)));
deadtimeFactor22 = (1 + parDeadTimeRef(1)*singlesRatePerBlock2 - nonParDeadTimeRef(1).*singlesRatePerBlock2.*singlesRatePerBlock2);
correctedSinglesRate2_DeadTimeCorrected1 = singlesRatePerRing2 ./ deadtimeFactor21;
correctedSinglesRate2_DeadTimeCorrected2 = singlesRatePerRing2 ./ deadtimeFactor22;
% Then correct for activity:
correctedSinglesRate2_DeadTimeCorrected1 = correctedSinglesRate2_DeadTimeCorrected1 ./ A2DecayFactor;
correctedSinglesRate2_DeadTimeCorrected2 = correctedSinglesRate2_DeadTimeCorrected2 ./ A2DecayFactor;
%% PLOT SINGLES RATES PER RING WITH DEAD TIME CORRECTION FOR METHOD 1
% Plot the singles rate with decay correction.
figure;
plot(1:structInterfile1.NumberOfRings, singlesRatePerRing1, 1:structInterfile1.NumberOfRings, correctedSinglesRate1_DeadTimeCorrected1, 1:structInterfile2.NumberOfRings, singlesRatePerRing2_DecayCorrected, ...
    1:structInterfile2.NumberOfRings, correctedSinglesRate2_DeadTimeCorrected1);
xlabel('Ring');
ylabel('Single Rates');
legend('Acquisition 1','Acquisition 1 - DT Correction', 'Acquisition 2 - Only Decay Correction','Acquisition 2 - Decay and DT correction');
title('Dead Time Correction with Equation 1 - Casey 1995');
set(gcf, 'position', [1 25 1920 1069]);
%% PLOT SINGLES RATES PER RING WITH DEAD TIME CORRECTION FOR METHOD 2
% Plot the singles rate with decay correction.
figure;
plot(1:structInterfile1.NumberOfRings, singlesRatePerRing1, 1:structInterfile1.NumberOfRings, correctedSinglesRate1_DeadTimeCorrected2, 1:structInterfile2.NumberOfRings, singlesRatePerRing2_DecayCorrected, ...
    1:structInterfile2.NumberOfRings, correctedSinglesRate2_DeadTimeCorrected2);
xlabel('Ring');
ylabel('Single Rates');
legend('Acquisition 1','Acquisition 1 - DT Correction', 'Acquisition 2 - Only Decay Correction','Acquisition 2 - Decay and DT correction');
title('Dead Time Correction with Equation 2 - Tai 1998');
set(gcf, 'position', [1 25 1920 1069]);
%% PLOT COUNTS IN DIRECT SINOGRAMS
% Plot the counts the direct sinograms, correcting for decay time and dead
% time.
directSinos1 = sinogram1(:,:,1:structInterfile1.NumberOfRings);
directSinos2 = sinogram2(:,:,1:structInterfile1.NumberOfRings);
% The delayed sinograms also change with the activity:
directDelayedSinos1 = delayedSinogram1(:,:,1:structInterfile1.NumberOfRings);
directDelayedSinos2 = delayedSinogram2(:,:,1:structInterfile1.NumberOfRings);

% Counts per second per sino:
CountsPerRing1 = sum(sum(directSinos1)) ./ structInterfile1.ImageDurationSec;
CountsPerRing1 = permute(CountsPerRing1, [1 3 2]);
CountsPerRing2 = sum(sum(directSinos2)) ./ structInterfile2.ImageDurationSec;
CountsPerRing2 = permute(CountsPerRing2, [1 3 2]);
% Idem for delays:
DelayedCountsPerRing1 = sum(sum(directDelayedSinos1)) ./ structInterfile1.ImageDurationSec;
DelayedCountsPerRing1 = permute(DelayedCountsPerRing1, [1 3 2]);
DelayedCountsPerRing2 = sum(sum(directDelayedSinos2)) ./ structInterfile2.ImageDurationSec;
DelayedCountsPerRing2 = permute(DelayedCountsPerRing2, [1 3 2]);

figure;
plot(1:structInterfile1.NumberOfRings, CountsPerRing1, 1:structInterfile1.NumberOfRings, DelayedCountsPerRing1, 1:structInterfile1.NumberOfRings, CountsPerRing2,...
    1:structInterfile1.NumberOfRings, DelayedCountsPerRing2);
xlabel('Ring');
ylabel('Cps');
legend('Acquisition 1', 'Delayed Acquisition 1', 'Acquisition 2', 'Delayed Acquisition 2');
title('Counts Per Second For Direct Sinograms of Prompt and Delayed Events of Two Different Acquisition');

% Take out the delayed events
CountsPerRing1 = CountsPerRing1 - DelayedCountsPerRing1;
CountsPerRing2 = CountsPerRing2 - DelayedCountsPerRing2;

% Decay Correction:
CountsPerRing2_DecayCorrected = CountsPerRing2 ./ A2DecayFactor;
% Dead Time for coincidence effRing1*effRing2 (for direct sinograms effRing²):
CountsPerRing1_DeadTimeCorrected1 = CountsPerRing1 ./ (deadtimeFactor11.*deadtimeFactor11);
CountsPerRing2_DeadTimeCorrected1 = CountsPerRing2 ./ (deadtimeFactor21.*deadtimeFactor21);
% the secnod need to be corrected for decay:
CountsPerRing2_DeadTimeCorrected1 = CountsPerRing2_DeadTimeCorrected1 ./ A2DecayFactor;
figure;
plot(1:structInterfile1.NumberOfRings, CountsPerRing1, 1:structInterfile1.NumberOfRings, CountsPerRing2, 1:structInterfile1.NumberOfRings, CountsPerRing2_DecayCorrected,...
    1:structInterfile1.NumberOfRings, CountsPerRing1_DeadTimeCorrected1, 1:structInterfile1.NumberOfRings, CountsPerRing2_DeadTimeCorrected1);
xlabel('Ring');
ylabel('Cps');
legend('Acquisition 1 Random Corrected', 'Acquisition 2 Random Corrected', 'Acquisition 2 - Only Random and Decay Correction','Acquisition 1 - Random and DT Correction','Acquisition 2 - Random, Decay and DT correction');
title('Dead Time Correction with Equation 1 - Casey 1995');
set(gcf, 'position', [1 25 1920 1069]);