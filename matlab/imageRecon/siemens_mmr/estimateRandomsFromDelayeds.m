% Estimate the randoms form the delayed with ML

function [randoms, structSizeSino] = estimateRandomsFromDelayeds(delayedSinogramSpan1, structSizeSino3dSpan1, normSinogramSpanN, structSizeSino3dSpanN, outputPath)
% This function creates the randoms estimate.
disp('estimate Randoms with Delayeds Sinogram...'); 
if ~isdir(outputPath)
    mkdir(outputPath);
end
timeWindows_nseg = 8;
% Check what OS I am running on:
if(strcmp(computer(), 'GLNXA64'))
    os = 'linux';
    pathBar = '/';
    sepEnvironment = ':';
elseif(strcmp(computer(), 'PCWIN') || strcmp(computer(), 'PCWIN64'))
    os = 'windows';
    pathBar = '\';
    sepEnvironment = ';';
else
    disp('OS not compatible');
    return;
end

% The input sinogram must be span 1:
if structSizeSino3dSpan1.span ~= 1
    error('error: The delayed sinogram must be span 1.'); 
end

% Method from Panin 2008. Maximices likelihood:
%L = sum(r_measured*log(r_estimated)-r_estimated);
%r_estimated=si*sj/2
% With Coordinate descent:
% s^n+1_k = s^n_k+sum(s^n_j*w*(r-r_measured) 

%% CREATE DETECTOR SYSTEM MATRIX
[detector1SystemMatrix, detector2SystemMatrix] = createDetectorSystemMatrix3d(1, 0);
combinedNormalization =  sum(detector1SystemMatrix',2) + sum(detector2SystemMatrix',2);
combSystemMatrix = (detector1SystemMatrix+detector2SystemMatrix);%./repmat(combinedNormalization',[size(detector1SystemMatrix,1) 1]);
% for i=1 : size(combSystemMatrix,2)
%     i
%     combSystemMatrix(i,:) = combSystemMatrix(i,:)./combinedNormalization';
% end
%% FAN-SUM OF DELAYEDS
singles = double(delayedSinogramSpan1(:))'*combSystemMatrix;
singles = sqrt(singles./combinedNormalization');
figure;
for i = 1 : 5
    % Create r_estimated:
    r_estimated = (detector1SystemMatrix*singles').*(detector2SystemMatrix*singles')./2;
    % Update singles:
    singles = singles + (detector1SystemMatrix'*double(delayedSinogramSpan1(:)-r_estimated))./(combSystemMatrix*(singles.^2));
    %% GENERATE RANDOM ESTIMATE FROM SINGLES
    randoms = timeWindows_nseg*1e-9*(singles*detector1SystemMatrix').*(singles*detector2SystemMatrix');
    randoms = reshape(randoms, [structSizeSino3dSpan1.numR structSizeSino3dSpan1.numTheta sum(structSizeSino3dSpan1.sinogramsPerSegment)]);
    structSizeSino = structSizeSino3dSpan1;

    plot([delaySinogramSpan11(:, 128,32) randomsSinogramSpan11(:, 128,32) normalizedRandomsSinogramSpan11(:, 128,32) ...
        randomsStir(:, 128,32) randomsFromDelayeds(:, 128,32)./mean(randomsFromDelayeds(:)).*mean(randomsStir(:))]);
end