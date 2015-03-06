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
%% READ SINOGRAMS
pathSinograms = '/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/13NCountRateSinograms/';
sinogramsNames = dir([pathSinograms '*uncomp.s.hdr']);
sinogramsNames(1:2,:) = [];
% Read them and get the info of the singles:
for i = 1 : numel(sinogramsNames)
    [structInterfile1{i}, structSizeSino] = getInfoFromSiemensIntf([pathSinograms sinogramsNames(i).name]);
    [sinograms{i}, delayedSinograms{i}, structSizeSino3d] = getIntfSinogramsFromUncompressedMmr([pathSinograms sinogramsNames(i).name(1:end-4)], [pathSinograms sinogramsNames(i).name(1:end-6) 'april']);
end

%% GET ALL THE NORM FILES IN THE PATH AND READ THEM
normFiles = dir([normPath '*.n']);
% Read the last normalization fie:
[componentFactors componentLabels] = readmMrComponentBasedNormalization([normPath normFiles(end).name], 0);

%% GENERATE THE STRUCT OF A SINO3D SPAN 11
% In that structure I have the amount of sinograms axially compressed per
% stored sinogram.
numR = 344; numTheta = 252; numZ = 64; maxAbsRingDiff = 60; rFov_mm = 594/2; zFov_mm = 258;
structSizeSino3dSpan11 = getSizeSino3dFromSpan(numR, numTheta, numZ, ...
    rFov_mm, zFov_mm, 11, maxAbsRingDiff);
%% GET PARALYZING AND NON PARALYZING DEAD TIME
parDeadTime = componentFactors{5};
nonParDeadTime = componentFactors{6};

%% CHECK TWO POSSIBLE FORMULAS
% 1) From Casey 1995:
singles = 0 : 1000 : 500000;
deadTimeEfficency1 = (exp(singles.*parDeadTime(1)./(1+singles.*nonParDeadTime(1)))./(1+singles.*nonParDeadTime(1)));
% 2) From Tai 1998:
deadTimeEfficency2 =  (1 + parDeadTime(1)*singles - nonParDeadTime(1).*singles.*singles);
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
deadTimeEfficency1 = (exp(singles.*parDeadTime(1)./(1+singles.*nonParDeadTime(1)))./(1+singles.*nonParDeadTime(1)));
% 2) From Tai 1998:
deadTimeEfficency2 =  (1 + parDeadTime(1)*singles - nonParDeadTime(1).*singles.*singles);
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
numberOfBuckets = structInterfile1{i}.NumberOfBuckets;
numberofAxialBlocks = 8;
numberofTransverseBlocksPerRing = 8;
numberOfBucketsInRing = numberOfBuckets / (numberofTransverseBlocksPerRing);
numberOfBlocksInBucketRing = numberOfBuckets / (numberofTransverseBlocksPerRing*numberOfAxialBlocksPerBucket);
numberOfTransverseCrystalsPerBlock = 9; % includes the gap
numberOfAxialCrystalsPerBlock = 8;
numberOfBucketRings = structInterfile1{i}.NumberOfRings / (numberOfAxialBlocksPerBucket*numberOfAxialCrystalsPerBlock);
% Get singles rate per ring for all the measurements:
singlesRatePerRing = cell(1, numel(sinogramsNames));
for i = 1 : numel(sinogramsNames)
    for j = 1 : numberOfBucketRings
        singlesRatePerRing{i}((j-1)*(numberOfAxialBlocksPerBucket*numberOfAxialCrystalsPerBlock)+1:j*(numberOfAxialBlocksPerBucket*numberOfAxialCrystalsPerBlock)) = ...
            sum(structInterfile1{i}.SinglesPerBucket((j-1)*numberOfBucketsInRing+1 : j*numberOfBucketsInRing)) ./ (numberOfAxialBlocksPerBucket*numberOfAxialCrystalsPerBlock);
    end
end
% Get singles rate per block for all the measurements:
singlesRatePerBlock = cell(1, numel(sinogramsNames));
for i = 1 : numel(sinogramsNames)
    for j = 1 : numberOfBucketRings
        singlesRatePerBlock{i}((j-1)*(numberOfAxialBlocksPerBucket*numberOfAxialCrystalsPerBlock)+1:j*(numberOfAxialBlocksPerBucket*numberOfAxialCrystalsPerBlock)) = ...
            mean(structInterfile1{i}.SinglesPerBucket((j-1)*numberOfBucketsInRing+1 : j*numberOfBucketsInRing)); % ./ (numberOfAxialBlocksPerBucket*numberOfTransverseBlocksPerBucket); % Using per bucket instead per block
    end
end
%% CORRECT FOR ACTIVITY DECAY
% Correct for activity decay.
% Half life N-13:
halfLife_min = 9.97;
halfLife_days = halfLife_min / (60*24);
% First acquisition.
initDateTimeNum = datenum(str2num(structInterfile1{1}.StudyDateYyyyMmDd(1:4)), str2num(structInterfile1{1}.StudyDateYyyyMmDd(6:7)), str2num(structInterfile1{1}.StudyDateYyyyMmDd(9:10)),...
    str2double(structInterfile1{1}.StudyTimeHhMmSsGmt0000(1:2)), str2double(structInterfile1{1}.StudyTimeHhMmSsGmt0000(4:5)), str2double(structInterfile1{1}.StudyTimeHhMmSsGmt0000(7:8)));

% Correct all the acuqisiton count rate for decay:
for i = 1 : numel(sinogramsNames)
    dateTimeNum2 = datenum(str2num(structInterfile1{i}.StudyDateYyyyMmDd(1:4)), str2num(structInterfile1{i}.StudyDateYyyyMmDd(6:7)), str2num(structInterfile1{i}.StudyDateYyyyMmDd(9:10)),...
        str2double(structInterfile1{i}.StudyTimeHhMmSsGmt0000(1:2)), str2double(structInterfile1{i}.StudyTimeHhMmSsGmt0000(4:5)), str2double(structInterfile1{i}.StudyTimeHhMmSsGmt0000(7:8)));
    timeBetween21_days(i) = dateTimeNum2 - initDateTimeNum;
    % Activity Decay Correction from measurement i to 1:
    A2DecayFactor = exp(-log(2)/halfLife_days.*timeBetween21_days(i));
    singlesRatePerRing_DecayCorrected{i} = singlesRatePerRing{i} ./ A2DecayFactor;
    singlesRatePerBlock_DecayCorrected{i} = singlesRatePerBlock{i} ./ A2DecayFactor;
end

%% PLOT SINGLES RATES PER RING WITHOUT DEAD TIME CORRECTION
% Convert the time to seconds:
timeBetween21_sec = timeBetween21_days * 60* 60 *24;
% Plot the singles rate with decay correction.
figure;
hold on;
for i = 1 : numel(sinogramsNames)
    plot(1:structInterfile1{i}.NumberOfRings, singlesRatePerRing{i}, 1:structInterfile1{i}.NumberOfRings, singlesRatePerRing_DecayCorrected{i});
end
xlabel('Ring');
ylabel('Single Rates');
%legend('Acquisition 1', sprintf('Acquisition 2 (%d days later)', daysBetween21), sprintf('Acquisition 2 Decay Corrected'));
set(gcf, 'position', [1 25 1920 1069]);

% Count Rate plot Per ring:
countRatesPerRing = zeros(numel(sinogramsNames), numberOfBucketRings);
countRatePerRing_DecayCorrected = zeros(numel(sinogramsNames), numberOfBucketRings);
for j = 1 : numberOfBucketRings 
    for i = 1 : numel(sinogramsNames)   
        countRatesPerRing(i,j) = singlesRatePerRing{i}((j-1)*numberOfAxialCrystalsPerBlock+1);
        countRatesPerBlock(i,j) = singlesRatePerBlock{i}((j-1)*numberOfAxialCrystalsPerBlock+1);
        countRatePerRing_DecayCorrected(i,j) = singlesRatePerRing_DecayCorrected{i}((j-1)*numberOfAxialCrystalsPerBlock+1);
        countRatePerBlock_DecayCorrected(i,j) = singlesRatePerBlock_DecayCorrected{i}((j-1)*numberOfAxialCrystalsPerBlock+1);
    end
    legendsPlot{j} = sprintf('Bucket Ring %d', j);
end
figure;
plot(timeBetween21_sec, countRatesPerRing);
legend(legendsPlot);
set(gcf, 'position', [1 25 1920 1069]);
xlabel('Time [sec]');
ylabel('Single Rates Per Ring');

figure;
plot(timeBetween21_sec, countRatePerRing_DecayCorrected);
legend(legendsPlot);
set(gcf, 'position', [1 25 1920 1069]);
xlabel('Time [sec]');
ylabel('Single Rates Per Ring Decay Corrected');
%% CORRECT FOR DEAD TIME
% Correct singles rate:
for i = 1 : numel(sinogramsNames)
    deadtimeFactors1 = (exp(countRatesPerBlock(i,:).*parDeadTime(1)./(1+countRatesPerBlock(i,:).*nonParDeadTime(1)))./(1+countRatesPerBlock(i,:).*nonParDeadTime(1)));
    correctedSinglesRatePerRing_DeadTimeCorrected1(i,:) = countRatesPerRing(i,:) ./ deadtimeFactors1;
    
    deadtimeFactors2 = (1 + parDeadTime(1)*countRatesPerBlock(i,:) - nonParDeadTime(1).*countRatesPerBlock(i,:).*countRatesPerBlock(i,:));
    correctedSinglesRatePerRing_DeadTimeCorrected2(i,:) = countRatesPerRing(i,:) ./ deadtimeFactors2;
end

% for i = 1 : numel(sinogramsNames)
%     deadtimeFactors1 = (exp(countRatesPerRing(i,:).*parDeadTime(1)./(1+countRatesPerRing(i,:).*nonParDeadTime(1)))./(1+countRatesPerRing(i,:).*nonParDeadTime(1)));
%     correctedSinglesRatePerRing_DeadTimeCorrected1(i,:) = countRatesPerRing(i,:) ./ deadtimeFactors1;
%     
%     deadtimeFactors2 = (1 + parDeadTime(1)*countRatesPerRing(i,:) - nonParDeadTime(1).*countRatesPerRing(i,:).*countRatesPerRing(i,:));
%     correctedSinglesRatePerRing_DeadTimeCorrected2(i,:) = countRatesPerRing(i,:) ./ deadtimeFactors2;
% end
%% ANALYSIS OF EACH BUCKET RING
% Get the measurement with less activity and get the singles rates expected
% for the previous measurements:
countRatesPerRingLowest = countRatesPerRing(end,:);
finalDateTimeNum = datenum(str2num(structInterfile1{end}.StudyDateYyyyMmDd(1:4)), str2num(structInterfile1{end}.StudyDateYyyyMmDd(6:7)), str2num(structInterfile1{end}.StudyDateYyyyMmDd(9:10)),...
    str2double(structInterfile1{end}.StudyTimeHhMmSsGmt0000(1:2)), str2double(structInterfile1{end}.StudyTimeHhMmSsGmt0000(4:5)), str2double(structInterfile1{end}.StudyTimeHhMmSsGmt0000(7:8)));
for i = 1 : numel(sinogramsNames)
    dateTimeNumI = datenum(str2num(structInterfile1{i}.StudyDateYyyyMmDd(1:4)), str2num(structInterfile1{i}.StudyDateYyyyMmDd(6:7)), str2num(structInterfile1{i}.StudyDateYyyyMmDd(9:10)),...
        str2double(structInterfile1{i}.StudyTimeHhMmSsGmt0000(1:2)), str2double(structInterfile1{i}.StudyTimeHhMmSsGmt0000(4:5)), str2double(structInterfile1{i}.StudyTimeHhMmSsGmt0000(7:8)));
    timeBetweenAcquisitionAndRef_days(i) = finalDateTimeNum - dateTimeNumI;
    % Activity Decay from measurement i to end:
    A2DecayFactor = exp(-log(2)/halfLife_days.*timeBetweenAcquisitionAndRef_days(i));
    expectedSinglesRates(i,:) = countRatesPerRingLowest ./ A2DecayFactor;
end

% One plot per ring:
figure;
for j = 1 : numberOfBucketRings 
    subplot(numberOfBucketRings/4, 4, j);
    plot(timeBetween21_sec, expectedSinglesRates(:,j), timeBetween21_sec, countRatesPerRing(:,j), timeBetween21_sec, correctedSinglesRatePerRing_DeadTimeCorrected1(:,j));
    legend('Expected Singles Rates', 'Detected Singles Rate', 'Detected Singles Rate Dead Time Corrected');
    title(sprintf('Bucket Ring %d', j));
    set(gcf, 'position', [1 25 1920 1069]);
    xlabel('Time [sec]');
    ylabel('Single Rates Per Ring');
end

% Plot the incident rate against the measures:
figure;
for j = 1 : numberOfBucketRings 
    subplot(numberOfBucketRings/4, 4, j);
    plot(expectedSinglesRates(:,j), countRatesPerRing(:,j), expectedSinglesRates(:,j), correctedSinglesRatePerRing_DeadTimeCorrected1(:,j));
    legend('Expected Singles Rates', 'Detected Singles Rate', 'Detected Singles Rate Dead Time Corrected');
    title(sprintf('Bucket Ring %d', j));
    set(gcf, 'position', [1 25 1920 1069]);
    xlabel('Incident Count Rate Ring');
    ylabel('Single Rates Per Ring');
    axis equal
end

%% PLOT COUNTS IN DIRECT SINOGRAMS
for i = 1 : numel(sinogramsNames)
    % Plot the counts the direct sinograms, correcting for decay time and dead
    % time.
    % Counts per second per sino:
    CountsPerSino = sum(sum(sum( sinograms{i}))) ./ structInterfile1{i}.ImageDurationSec;
    TotalCoincidenceRate(i) = permute(CountsPerSino, [1 3 2]);
    CountsPerSino = sum(sum(sum( delayedSinograms{i}))) ./ structInterfile1{i}.ImageDurationSec;
    TotalRandomsRate(i) = permute(CountsPerSino, [1 3 2]);
end

for i = 1 : numel(sinogramsNames)
    % Plot the counts the direct sinograms, correcting for decay time and dead
    % time.
    % Counts per second per sino:
    CountsPerRingAux = sum(sum( sinograms{i}(:,:,1:structInterfile1{i}.NumberOfRings))) ./ structInterfile1{i}.ImageDurationSec;
    CoincidenceRatePerRing(i,:) = permute(CountsPerRingAux, [1 3 2]);
end


figure;
plot(timeBetween21_sec, TotalCoincidenceRate, timeBetween21_sec, TotalRandomsRate);
xlabel('Time [sec]');
ylabel('Cps');
title('Total Coincidence Rate');
legend('Prompt Coincidence Rate', 'Randoms Coincidence Rate');
set(gcf, 'position', [1 25 1920 1069]);

figure;
hold on;
for i = 1 : size(CoincidenceRatePerRing,2)
    plot([1:size(CoincidenceRatePerRing,1)]', CoincidenceRatePerRing(:,i));
    lengends{i} = sprintf('Ring %d', i);
end
xlabel('Time [sec]');
ylabel('Cps');
legend(lengends);
title('Coincide Rate Per Ring');
set(gcf, 'position', [1 25 1920 1069]);