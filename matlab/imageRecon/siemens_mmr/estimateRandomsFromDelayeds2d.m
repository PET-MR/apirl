% Estimate the randoms form the delayed with ML

function [randoms, singles] = estimateRandomsFromDelayeds2d(delayedSinogram2d, structSizeSino2d, numIterations)


% This function creates the randoms estimate.
disp('estimate Randoms with Delayeds Sinogram...'); 

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


% Method from Panin 2008. Maximices likelihood:
%L = sum(r_measured*log(r_estimated)-r_estimated);
%r_estimated=si*sj/2
% With Coordinate descent:
% s^n+1_k = s^n_k+sum(s^n_j*w*(r-r_measured) 

%% CREATE DETECTOR SYSTEM MATRIX
disp('Inicialization of Crystal-Sinogram Matrix...');
[detector1SystemMatrix, detector2SystemMatrix] = createDetectorSystemMatrix2d(1, 0);
detector1Normalization = sum(detector1SystemMatrix',2)';
detector2Normalization = sum(detector2SystemMatrix',2)';
combinedNormalization =  sum(detector1SystemMatrix',2) + sum(detector2SystemMatrix',2);
combSystemMatrix = (detector1SystemMatrix+detector2SystemMatrix);%./repmat(combinedNormalization',[size(detector1SystemMatrix,1) 1]);
% for i=1 : size(combSystemMatrix,2)
%     i
%     combSystemMatrix(i,:) = combSystemMatrix(i,:)./combinedNormalization';
% end
% [detector1SystemMatrix, detector2SystemMatrix] = createDetectorSystemMatrix2d(1, 0);
% combinedNormalization =  sum(detector1SystemMatrix',2) + sum(detector2SystemMatrix',2);
% combSystemMatrix = (detector1SystemMatrix+detector2SystemMatrix);%./repmat(combinedNormalization',[size(detector1SystemMatrix,1) 1]);

%% GET A RANDOMS SINOGRAM FOR EACH RING
% They are independent because its 2d:
for i = 1 : structSizeSino2d.numZ
    delayedSinogram2d_ring_i = delayedSinogram2d(:,:,i);
    %% FAN-SUM OF DELAYEDS
    disp('Computing initial estimate...');
    fansum = combSystemMatrix'*double(delayedSinogram2d_ring_i(:));
    singles = sqrt(fansum./combinedNormalization);
    % randoms = (detector1SystemMatrix*double(singles)).*(detector2SystemMatrix*double(singles));
    % randoms = reshape(randoms, [structSizeSino3dSpan1.numR structSizeSino3dSpan1.numTheta sum(structSizeSino3dSpan1.sinogramsPerSegment)]);
    % sum((delayedSinogramSpan1(:)-randoms(:)).^2)

    for j = 1 : numIterations
        disp(sprintf('Iteration %d for Ring %d...',j,i));
        %% GENERATE RANDOM ESTIMATE FROM SINGLES
        aux1= combSystemMatrix'*double(delayedSinogram2d_ring_i(:));
        aux3 = detector1SystemMatrix'*(detector2SystemMatrix*double(singles));
        aux4 = detector2SystemMatrix'*(detector1SystemMatrix*double(singles));
        singles = (aux1)./((aux3+aux4));
    end
    randoms_ring_i = (detector1SystemMatrix*double(singles)).*(detector2SystemMatrix*double(singles));
    randoms(:,:,i) = reshape(randoms_ring_i, [structSizeSino2d.numR structSizeSino2d.numTheta]);
end

    
