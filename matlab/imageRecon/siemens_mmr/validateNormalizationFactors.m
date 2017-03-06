%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 20/10/2015
%  *********************************************************************
%  Validates the normalization factors comparing with the one expanded with
%  e7 tools.
clear all 
close all

apirlPath = [fileparts(mfilename('fullpath')) filesep '..' filesep '..' filesep '..'];
%% CUDA PATH
cudaPath = '/usr/local/cuda/';
setenv('PATH', [getenv('PATH') pathsep cudaPath filesep 'bin']);
setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') pathsep cudaPath filesep 'lib64']);
%% STIR PATH
stirPath = '/usr/local/stir3.0/';
stirMatlabPath = '/home/mab15/workspace/KCL/apirl-kcl/trunk/stir/';
scriptsPath = [stirMatlabPath 'scripts/'];
%% SET ENVIRONMENT AND MATLAB PATHS
addpath(genpath([apirlPath filesep 'matlab']));
addpath(genpath(stirMatlabPath));
setenv('PATH', [getenv('PATH') pathsep apirlPath filesep 'build' filesep 'bin' pathsep stirPath filesep 'bin/']);
setenv('LD_LIBRARY_PATH', [getenv('LD_LIBRARY_PATH') pathsep apirlPath filesep 'build' filesep 'bin' pathsep stirPath filesep 'lib/' ]);
%% ACQUISITION FILES
sinogramFilename = '/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/BRAIN_PETMR/SINOGRAMS/PET_ACQ_68_20150610155347-0uncomp.s.hdr';
normFilename = '/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/BRAIN_PETMR/norm/Norm_20150609084317.n';
attMapBaseFilename = '/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/BRAIN_PETMR/SINOGRAMS/PET_ACQ_68_20150610155347';
pixelSize_mm = [2.08625 2.08625 2.03125];
%% READ EMISSION HEADER TO GET SINGLES PER BUCKET
[info, structSizeSino] = getInfoFromSiemensIntf(sinogramFilename);
singles_rates_per_bucket = info.SinglesPerBucket;
%% SINOGRAM'S SIZE
% Size of mMr Sinogram's
numTheta = 252; numR = 344; numRings = 64; maxAbsRingDiff = 60; rFov_mm = 594/2; zFov_mm = 258;
span_choice = 1;
structSizeSino3d_span1 = getSizeSino3dFromSpan(numR, numTheta, numRings, rFov_mm, zFov_mm, span_choice, maxAbsRingDiff);
span_choice = 11;
structSizeSino3d_span11 = getSizeSino3dFromSpan(numR, numTheta, numRings, rFov_mm, zFov_mm, span_choice, maxAbsRingDiff);
%% E7 TOOLS NCF
ncf_e7_filename = '/home/mab15/workspace/KCL/Biograph_mMr/Mediciones/BRAIN_PETMR/e7_tools/tmp/norm3d_00.a';
fid = fopen(ncf_e7_filename, 'r');
if fid == -1
    error('Error reading the normalization correction factors from the e7 tools.');
end
ncf_e7 = fread(fid, structSizeSino3d_span11.numR*structSizeSino3d_span11.numTheta*sum(structSizeSino3d_span11.sinogramsPerSegment), 'single');
ncf_e7 = reshape(ncf_e7, [structSizeSino3d_span11.numR structSizeSino3d_span11.numTheta sum(structSizeSino3d_span11.sinogramsPerSegment)]);
fclose(fid);
%% GENERATE NCFs
% Complete NCFs Span1 and time-invarian NCFs (axial siemens factor included for span11):
[overall_ncf_3d_span1, scanner_time_invariant_ncf_3d_span1, scanner_time_variant_ncf_3d_span1, acquisition_dependant_ncf_3d_span1, used_xtal_efficiencies_span1, used_deadtimefactors_span1, used_axial_factors_span1] = ...
   create_norm_files_mmr(normFilename, [], [], [], singles_rates_per_bucket, structSizeSino3d_span1.span);
% Complete NCFs Span11:
span = structSizeSino3d_span11.span; % Fixed axial factors.
span = []; % time-variant axial factors.
[overall_ncf_3d_span11, scanner_time_invariant_ncf_3d_span11, scanner_time_variant_ncf_3d_span11, acquisition_dependant_ncf_3d_span11, used_xtal_efficiencies_span11, used_deadtimefactors_span11, used_axial_factors_span11] = ...
   create_norm_files_mmr(normFilename, [], [], [], singles_rates_per_bucket, span);
%% WRITE IN FOLDER
outputPath = '/home/mab15/workspace/KCL/Biograph_mMr/Normalization/validation/';
if ~isdir(outputPath)
    mkdir(outputPath);
end
% Write e7:
interfileWriteSino(single(ncf_e7), [outputPath 'e7_ncf'], structSizeSino3d_span11);
% Write mine:
interfileWriteSino(single(overall_ncf_3d_span11), [outputPath 'apirl_ncf'], structSizeSino3d_span11);
interfileWriteSino(single(scanner_time_invariant_ncf_3d_span11), [outputPath 'apirl_ti_ncf'], structSizeSino3d_span11);
interfileWriteSino(single(scanner_time_variant_ncf_3d_span11), [outputPath 'apirl_tv_ncf'], structSizeSino3d_span11);
interfileWriteSino(single(acquisition_dependant_ncf_3d_span11), [outputPath 'apirl_acq_ncf'], structSizeSino3d_span11);
%% PLOT AXIAL PROFILES
ncf_e7_with_gaps =ncf_e7;
ncf_e7_with_gaps(overall_ncf_3d_span11 == 0) = 0;
axial_e7_tools = permute(sum(sum(ncf_e7_with_gaps)), [3 1 2]);
axial_apirl = permute(sum(sum(overall_ncf_3d_span11)), [3 1 2]);
indexSinos = 1 : sum(structSizeSino3d_span11.sinogramsPerSegment);
figure;
plot(indexSinos, axial_e7_tools, indexSinos, axial_apirl);
%% CHECK DEAD TIME
% Use a standard math stored with apirl:
[componentFactors, componentLabels]  = readmMrComponentBasedNormalization(normFilename, 0);
parDeadTime = componentFactors{5};
nonParDeadTime = componentFactors{6};
% Parameters of the buckets:
numberOfTransverseBlocksPerBucket = 2;
numberOfAxialBlocksPerBucket = 1;
numberOfBuckets = 224;
numberofAxialBlocks = 8;
numberofTransverseBlocksPerRing = 8;
numberOfBucketsInRing = numberOfBuckets / (numberofTransverseBlocksPerRing);
numberOfBlocksInBucketRing = numberOfBuckets / (numberofTransverseBlocksPerRing*numberOfAxialBlocksPerBucket);
numberOfTransverseCrystalsPerBlock = 9; % includes the gap
numberOfAxialCrystalsPerBlock = 8;
numberOfAxialCrystalsPerBucket = numberOfAxialCrystalsPerBlock*numberOfAxialBlocksPerBucket;

% Get the single rate per ring:
singles_rates_per_ring = singles_rates_per_bucket / (numberOfAxialBlocksPerBucket*numberOfAxialCrystalsPerBlock);
% Compute dead time factors, is equivalent to an efficency factor.
% Thus, a factor for each crystal unit is computed, and then both
% factors are multiplied.
% Detector Ids:
[map2dDet1Ids, map2dDet2Ids] = createMmrDetectorsIdInSinogram();
% BucketId:
map2dBucket1Ids = ceil(map2dDet1Ids/(numberOfTransverseBlocksPerBucket*numberOfTransverseCrystalsPerBlock));
map2dBucket2Ids = ceil(map2dDet2Ids/(numberOfTransverseBlocksPerBucket*numberOfTransverseCrystalsPerBlock));
% Replicate for 3d map:
indiceSino = 1; % indice del sinogram 3D.
for segment = 1 : structSizeSino3d_span1.numSegments
    % Por cada segmento, voy generando los sinogramas correspondientes y
    % contándolos, debería coincidir con los sinogramas para ese segmento: 
    numSinosThisSegment = 0;
    % Recorro todos los z1 para ir rellenando
    for z1 = 1 : (structSizeSino3d_span1.numZ*2)
        numSinosZ1inSegment = 0;   % Cantidad de sinogramas para z1 en este segmento
        % Recorro completamente z2 desde y me quedo con los que están entre
        % minRingDiff y maxRingDiff. Se podría hacer sin recorrer todo el
        % sinograma pero se complica un poco.
        z1_aux = z1;    % z1_aux la uso para recorrer.
        for z2 = 1 : structSizeSino3d_span1.numZ
            % Ahora voy avanzando en los sinogramas correspondientes,
            % disminuyendo z1 y aumentnado z2 hasta que la diferencia entre
            % anillos llegue a maxRingDiff.
            if ((z1_aux-z2)<=structSizeSino3d_span1.maxRingDiff(segment))&&((z1_aux-z2)>=structSizeSino3d_span1.minRingDiff(segment))
                % Me asguro que esté dentro del tamaño del michelograma:
                if(z1_aux>0)&&(z2>0)&&(z1_aux<=structSizeSino3d_span1.numZ)&&(z2<=structSizeSino3d_span1.numZ)
                    numSinosZ1inSegment = numSinosZ1inSegment + 1;
                    % Get the detector id for the sinograms. The 2d id I
                    % need to add the offset of the ring:
                    map3dBucket1Ids(:,:,indiceSino) = map2dBucket1Ids + floor((z1_aux-1)/numberOfAxialCrystalsPerBucket) *numberOfBucketsInRing;
                    map3dBucket2Ids(:,:,indiceSino) = map2dBucket2Ids + floor((z2-1)/numberOfAxialCrystalsPerBucket)*numberOfBucketsInRing;
                end
            end
            % Pase esta combinación de (z1,z2), paso a la próxima:
            z1_aux = z1_aux - 1;
        end
        if(numSinosZ1inSegment>0)
            numSinosThisSegment = numSinosThisSegment + 1;
            indiceSino = indiceSino + 1;
        end
    end    
end

% Get the dead time for each detector and then multipliy to get the one
% for the bin.
deadTimeFactorsD1 = (1 + parDeadTime(1)*singles_rates_per_bucket(map3dBucket1Ids)/2 - nonParDeadTime(1).*singles_rates_per_bucket(map3dBucket1Ids).*singles_rates_per_bucket(map3dBucket1Ids)/4);
deadTimeFactorsD2 = (1 + parDeadTime(1)*singles_rates_per_bucket(map3dBucket2Ids)/2 - nonParDeadTime(1).*singles_rates_per_bucket(map3dBucket2Ids).*singles_rates_per_bucket(map3dBucket2Ids)/4);
acquisition_dependant_ncf_3d = 1./(deadTimeFactorsD1 .* deadTimeFactorsD2);
[acquisition_dependant_ncf_3d_span11, structSizeSino3d_span11] = convertSinogramToSpan(acquisition_dependant_ncf_3d, structSizeSino3d_span1, 11);
for i = 1 : sum(structSizeSino3d_span11.sinogramsPerSegment)
    % Normalize
    acquisition_dependant_ncf_3d_span11(:,:,i) = acquisition_dependant_ncf_3d_span11(:,:,i) ./ structSizeSino3d_span11.numSinosMashed(i);
end
interfileWriteSino(single(acquisition_dependant_ncf_3d_span11), [outputPath 'apirl_acq_ncf'], structSizeSino3d_span11);
%%
figure;
indexR = 5;
indexSino = 2;
indexProfiles = 1 : size(ncf_e7_with_gaps,2);
plot(indexProfiles, ncf_e7_with_gaps(indexR,:,indexSino)./overall_ncf_3d_span11(indexR,:,indexSino), indexProfiles,  acquisition_dependant_ncf_3d_span11(indexR,:,indexSino));
%%
% %% ONLY CRYSTALS
% % Use a standard math stored with apirl:
% [componentFactors, componentLabels]  = readmMrComponentBasedNormalization(normFilename, 0);
% % Use the crystal efficencies of the .n file:
% used_xtal_efficiencies = componentFactors{3};
% % Include the gaps:
% %used_xtal_efficiencies(1:9:end,:) = 0;
% 
% % 5) We generate the scan_dependent_ncf_3d, that is compound by the
% % dead-time and crystal-efficencies. We do it in span 1, and then after
% % including the dead time we compress to the wanted span.
% % a) Crystal efficencies. A sinogram is generated from the crystal
% % efficencies:
% scanner_time_variant_ncf_3d_span11 = createSinogram3dFromDetectorsEfficency(used_xtal_efficiencies, structSizeSino3d_span11, 0);
% % the ncf is 1/efficency:
% nonzeros = scanner_time_variant_ncf_3d_span11 ~= 0;
% scanner_time_variant_ncf_3d_span11(nonzeros) = 1./ scanner_time_variant_ncf_3d_span11(nonzeros);
% %%
% numDetectors = 504;
% detectorIds = 1 : numDetectors;
% theta = [0:structSizeSino3d_span11.numTheta-1]'; % The index of thetas goes from 0 to numTheta-1 (in stir)
% r = (-structSizeSino3d_span11.numR/2):(-structSizeSino3d_span11.numR/2+structSizeSino3d_span11.numR-1);
% [THETA, R] = meshgrid(theta,r);
% mapaDet1Ids = rem((THETA + floor((R)/2) + numDetectors)-1, numDetectors) + 1;   % The +1 is added in matlab version respect than c version, because here we have 1-base indexes.
% mapaDet2Ids = rem((THETA - floor((R+1)/2) + numDetectors/2 - 1), numDetectors) + 1; % The +1 is added in matlab version respect than c version, because here we have 1-base indexes.
% histDetIds = hist([mapaDet1Ids(:); mapaDet2Ids(:)], detectorIds);
% z1_aux = 1; z2 =1;
% efficenciesZ1 = used_xtal_efficiencies(:,z1_aux);
% efficenciesZ2 = used_xtal_efficiencies(:,z2);
% scanner_time_variant_ncf_3d_span11(:,:,1) = 1./(efficenciesZ1(mapaDet1Ids) .* efficenciesZ2(mapaDet2Ids));
% %% PLOT PROFILES
% indexR = 1;
% indexSino = 1;
% figure;
% indexProfiles = 1 : size(ncf_e7_with_gaps,2);
% ratio = ncf_e7(indexR,:,indexSino)./scanner_time_invariant_ncf_3d_span11(indexR,:,indexSino);
% ratio(scanner_time_invariant_ncf_3d_span11(indexR,:,indexSino)==0) = 0;
% plot(indexProfiles, ratio, indexProfiles, scanner_time_variant_ncf_3d_span11(indexR,:,indexSino));
% hold on;
% plot(indexProfiles, 1./efficenciesZ1(mapaDet1Ids(indexR,:,indexSino)), indexProfiles,  1./efficenciesZ2(mapaDet2Ids(indexR,:,indexSino)));
% %%
% indexR = 21;
% indexSino = 1;
% figure;
% indexProfiles = 1 : size(ncf_e7_with_gaps,2);
% ratio = ncf_e7(indexR,:,indexSino)./scanner_time_invariant_ncf_3d_span11(indexR,:,indexSino);
% ratio(scanner_time_invariant_ncf_3d_span11(indexR,:,indexSino)==0) = 0;
% plot(indexProfiles, ratio, indexProfiles, scanner_time_variant_ncf_3d_span11(indexR,:,indexSino));
% figure
% plot(indexProfiles, ratio, indexProfiles, 1./(efficenciesZ1(mapaDet1Ids(indexR,:,indexSino)).*efficenciesZ2(mapaDet2Ids(indexR,:,indexSino))),indexProfiles, 1./efficenciesZ1(mapaDet1Ids(indexR,:,indexSino)), indexProfiles,  1./efficenciesZ2(mapaDet2Ids(indexR,:,indexSino)));
% legend('e7', 'apirl', 'd1', 'd2');
% %%
% indexR = 1;
% indexSino = 574;
% figure;
% indexProfiles = 1 : size(ncf_e7_with_gaps,2);
% ratio = ncf_e7(indexR,:,indexSino)./scanner_time_invariant_ncf_3d_span11(indexR,:,indexSino);
% ratio(scanner_time_invariant_ncf_3d_span11(indexR,:,indexSino)==0) = 0;
% plot(indexProfiles, ratio, indexProfiles, scanner_time_variant_ncf_3d_span11(indexR,:,indexSino));