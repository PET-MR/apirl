%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 09/02/2015
%  *********************************************************************
%  function [overall_ncf_3d, scan_independent_ncf_3d, scan_dependent_ncf_3d, used_xtal_efficiencies, used_deadtimefactors] = ...
%       create_norm_files_mmr(cbn_filename, my_selection_of_xtal_efficiencies, my_choice_of_deadtimefactors, singles_rates_per_bucket, span_choice)
% 
%  This functions read the component based normalization file of the mMr
%  and creates the normalization factors for 3d sinograms with a
%  configurable span. Optionally it can use externals crystal efficencies.
%  Receives the following parameters:
%   -cbn_filename: the name of the file with the component based normalization (*.n).
%   -my_selection_of_xtal_efficiencies: this is a single precision 2D float data array of dimensions (504, 64), here 504 is the number of crystal
%       elements in a ring, and 64 is the number of rings. This parameter
%       is optional and must be used if the user wants to use different
%       crystal efficencies than the ones available in the .n file. If you
%       want to stick with the efficencies of the .n file, just use an
%       empty vector in this parameter [].
%   -my_choice_of_deadtimefactors: this is an optinal parameter. If you
%       want to use different dead time factors from the ones in the .n files.
%       If not just use an empty vector for this parameter []. It's a
%       matrix of numRings x 2. The first column is paralyzing dead time
%       and the second column non-paralyzing.
%   -singles_rates_per_bucket: singles rate per bucket used to estimate
%       dead time factors. Its an array of 228 elements (number of buckets in
%       the whole scanner). The order is the same than in the interfile
%       sinogram of an acquisition. If empty the dead time is not used.
%   -span_choice: span of the 3d sinogram. At the moment it's only
%       available for span 11. 
%
%  The function returns de following outputs:
%   -overall_ncf_3d: overall normalization correction factors. Its the
%       multiplication of the scan_independent_ncf_3d with the
%       scan_dependent_ncf_3d.
%   -scan_independent_ncf_3d: normalization correction factors without scan
%       dependant factors (dead time and crystal efficencies).
%   -scan_dependent_ncf_3d: normalization correction factors for scan
%       dependant factors (dead-time and crystal efficencies).
%   -used_xtal_efficiencies: crystal effincecies factors used in the
%       overall_ncf_3d. 
%   -used_deadtimefactors: dead time contstants used for compute the dead
%   time factors.
% 
%  The size of each component matrix are hardcoded for the mMr scanner and
%  are

function [overall_ncf_3d, scanner_time_invariant_ncf_3d, scanner_time_variant_ncf_3d, acquisition_dependant_ncf_3d, crystal_dependant_ncf_3d, used_xtal_efficiencies, used_deadtimefactors, used_axial_factors, structSizeSino3d] = ...
   create_norm_files_mmr(cbn_filename, my_axial_factors, my_selection_of_xtal_efficiencies, my_choice_of_deadtimefactors, singles_rates_per_bucket, span_choice)


used_xtal_efficiencies = [];
used_deadtimefactors = [];
used_axial_factors = [];

% 1) Read the .n files and get each component in a cell array:
if ~isempty(cbn_filename)
    [componentFactors, componentLabels]  = readmMrComponentBasedNormalization(cbn_filename, 0);
else
    % Use a standard math stored with apirl:
    [componentFactors, componentLabels]  = readmMrComponentBasedNormalization('exampleBinaryNormFileMmr.n', 0);
end

% 2) Size of the mmr's sinograms. The 3d parameters are generated from the
% selected span:
% Size of mMr Sinogram's
numTheta = 252; numR = 344; numRings = 64; maxAbsRingDiff = 60; rFov_mm = 594/2; zFov_mm = 258;
if isempty(span_choice)
    % If empty the default span 11.
    structSizeSino3d = getSizeSino3dFromSpan(numR, numTheta, numRings, rFov_mm, zFov_mm, 11, maxAbsRingDiff);
else
    structSizeSino3d = getSizeSino3dFromSpan(numR, numTheta, numRings, rFov_mm, zFov_mm, span_choice, maxAbsRingDiff);
end
structSizeSino3d_span1 = getSizeSino3dFromSpan(numR, numTheta, numRings, rFov_mm, zFov_mm, 1, maxAbsRingDiff);
% Total number of singorams per 3d sinogram:
numSinograms = sum(structSizeSino3d.sinogramsPerSegment);
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
% Generate the sinograms:
overall_ncf_3d = zeros(numR, numTheta, numSinograms, 'single');
scanner_time_invariant_ncf_3d = zeros(numR, numTheta, numSinograms, 'single');
scanner_time_variant_ncf_3d = zeros(numR, numTheta, numSinograms, 'single');
acquisition_dependant_ncf_3d = zeros(numR, numTheta, numSinograms, 'single');

% 3) Selection of dead-time parameters and crystal efficencies:
if isempty(my_selection_of_xtal_efficiencies)
    % Use the crystal efficencies of the .n file:
    used_xtal_efficiencies = componentFactors{3};
    % Include the gaps:
    %used_xtal_efficiencies(1:9:end,:) = 0;
    used_xtal_efficiencies(9:9:end,:) = 0;
else
    % Use the crystal efficencies received in the parameter file:
    used_xtal_efficiencies = my_selection_of_xtal_efficiencies;
    % Include the gaps:
    %used_xtal_efficiencies(1:9:end,:) = 0;
    used_xtal_efficiencies(9:9:end,:) = 0;
end

if isempty(my_choice_of_deadtimefactors)
    % Use the deadtime of the .n file. We hace to compute the sinogram with
    % the paralyzed and non-paralyzed dead time factors:
    paralyzed_ring_dead_time = componentFactors{5};
    non_paralyzed_ring_dead_time = componentFactors{6};
    used_deadtimefactors = [paralyzed_ring_dead_time non_paralyzed_ring_dead_time];
else
    % Use the deadtime efficencies received in the parameter file:
    used_deadtimefactors = my_choice_of_deadtimefactors;
end

% 4) We generate the scan_independent_ncf_3d, that is compound by the
% geometricFactor, the crystal interference and the axial effects:
% a) Geometric Factor. The geomtric factor is one projection profile per
% plane. But it's the same for all planes, so I just use one of them.
geometricFactor = repmat(single(componentFactors{1}(:,1)), 1, structSizeSino3d.numTheta);
% b) Crystal interference, its a pattern that is repeated peridoically:
crystalInterfFactor = single(componentFactors{2});
crystalInterfFactor = repmat(crystalInterfFactor', 1, structSizeSino3d.numTheta/size(crystalInterfFactor,1));
% c) Axial factors:
if isempty(my_axial_factors)
    axialFactors = zeros(sum(structSizeSino3d_span1.sinogramsPerSegment),1);
    axialFactorsPerRing = zeros(64,64);
    if isempty(span_choice)
        axialFactors = structSizeSino3d.numSinosMashed'.* (1./(componentFactors{4}));    % e7 tools does not use the other component factors.*componentFactors{8}));
    else
        % Generate axial factors from a block profile saved before:
        gainPerPixelInBlock = load('axialGainPerPixelInBlock.mat');
        % Check if the sensitivity profle is available:
        if exist('sensitivityPerSegment.mat') ~= 0
            sensitivityPerSegment = load('sensitivityPerSegment.mat');
            sensitivityPerSegment = sensitivityPerSegment.sensitivityPerSegment;
        else
            sensitivityPerSegment = ones(structSizeSino3d_span1.numSegments,1);
        end
        if(isstruct(gainPerPixelInBlock))
            gainPerPixelInBlock = gainPerPixelInBlock.gainPerPixelInBlock;
        end
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
                            % Get the index of detector inside a block:
                            pixelInBlock1 = rem(z1_aux-1, numberOfAxialCrystalsPerBlock);
                            pixelInBlock2 = rem(z2-1, numberOfAxialCrystalsPerBlock);
                            axialFactorsPerRing(z1_aux,z2) = (gainPerPixelInBlock(pixelInBlock1+1) * gainPerPixelInBlock(pixelInBlock2+1));
                            axialFactors(indiceSino) = axialFactors(indiceSino) + (gainPerPixelInBlock(pixelInBlock1+1) * gainPerPixelInBlock(pixelInBlock2+1));
                        end
                    end
                    % Pase esta combinación de (z1,z2), paso a la próxima:
                    z1_aux = z1_aux - 1;
                end
                if(numSinosZ1inSegment>0)
                    % I average the efficencies dividing by the number of axial
                    % combinations used for this sino:
                    %acquisition_dependant_ncf_3d(:,:,indiceSino) = acquisition_dependant_ncf_3d(:,:,indiceSino) / numSinosZ1inSegment;
                    numSinosThisSegment = numSinosThisSegment + 1;
                    indiceSino = indiceSino + 1;
                end
            end    
        end
        axialFactors_xtal_dependent = axialFactors;
        %% Normalize to ones:
        %axialFactors = axialFactors ./ mean(axialFactors);
        
        %% Sensitivity factors:
%         sensitivityFactors = zeros(sum(structSizeSino3d.sinogramsPerSegment),1);
%         for segment = 1 : structSizeSino3d.numSegments
%             indices = (structSizeSino3d_span1.minRingDiff>=structSizeSino3d.minRingDiff(segment))&(structSizeSino3d_span1.minRingDiff<=structSizeSino3d.maxRingDiff(segment));
%             meanValuePerSegment(segment) = mean(sensitivityPerSegment(indices));
%         end
%         indiceInicioSino = 1;
%         for segment = 1 : numel(structSizeSino3d.sinogramsPerSegment)
%             indiceFinSino = indiceInicioSino + structSizeSino3d.sinogramsPerSegment(segment) - 1;
%             sensitivityFactors(indiceInicioSino:indiceFinSino) = meanValuePerSegment(segment);
%             indiceInicioSino = indiceFinSino + 1;
%         end
%         sensitivityFactors = sensitivityFactors ./ mean(sensitivityFactors);
%         
%         % Apply sinogram normaliztion (in this case it is not necessary, but fo
%         %r other spans it is:
%         %axialFactors = axialFactors.*structSizeSino3d.numSinosMashed'; % All ones!
%         
%         axialFactors= axialFactors.*sensitivityFactors;
        % compress:
        if span_choice ~= 1
            axialFactorsSpan1 = axialFactors;
            axialFactors = zeros(sum(structSizeSino3d.sinogramsPerSegment),1);
            indiceSino = 1; % indice del sinogram 3D.
            for segment = 1 : structSizeSino3d.numSegments
                % Por cada segmento, voy generando los sinogramas correspondientes y
                % contándolos, debería coincidir con los sinogramas para ese segmento: 
                numSinosThisSegment = 0;
                % Recorro todos los z1 para ir rellenando
                for z1 = 1 : (structSizeSino3d.numZ*2)
                    numSinosZ1inSegment = 0;   % Cantidad de sinogramas para z1 en este segmento
                    % Recorro completamente z2 desde y me quedo con los que están entre
                    % minRingDiff y maxRingDiff. Se podría hacer sin recorrer todo el
                    % sinograma pero se complica un poco.
                    z1_aux = z1;    % z1_aux la uso para recorrer.
                    for z2 = 1 : structSizeSino3d.numZ
                        % Ahora voy avanzando en los sinogramas correspondientes,
                        % disminuyendo z1 y aumentnado z2 hasta que la diferencia entre
                        % anillos llegue a maxRingDiff.
                        if ((z1_aux-z2)<=structSizeSino3d.maxRingDiff(segment))&&((z1_aux-z2)>=structSizeSino3d.minRingDiff(segment))
                            % Me asguro que esté dentro del tamaño del michelograma:
                            if(z1_aux>0)&&(z2>0)&&(z1_aux<=structSizeSino3d.numZ)&&(z2<=structSizeSino3d.numZ)
                                numSinosZ1inSegment = numSinosZ1inSegment + 1;
                                axialFactors(indiceSino) = axialFactors(indiceSino) + axialFactorsPerRing(z1_aux,z2);
                            end
                        end
                        % Pase esta combinación de (z1,z2), paso a la próxima:
                        z1_aux = z1_aux - 1;
                    end
                    if(numSinosZ1inSegment>0)
                        % I average the efficencies dividing by the number of axial
                        % combinations used for this sino:
                        %acquisition_dependant_ncf_3d(:,:,indiceSino) = acquisition_dependant_ncf_3d(:,:,indiceSino) / numSinosZ1inSegment;
                        numSinosThisSegment = numSinosThisSegment + 1;
                        indiceSino = indiceSino + 1;
                    end
                end    
            end
        end
    end
else
    axialFactors = my_axial_factors;
end

% Generate scanner time invariant:
for i = 1 : sum(structSizeSino3d.sinogramsPerSegment)
    % First the geomeitric, crystal interference factors:
    scanner_time_invariant_ncf_3d(:,:,i) = 1./(geometricFactor .* crystalInterfFactor);
    % Axial factor:
    scanner_time_invariant_ncf_3d(:,:,i) = scanner_time_invariant_ncf_3d(:,:,i) .* (1./axialFactors(i));
end

% 5) We generate the scan_dependent_ncf_3d, that is compound by the
% dead-time and crystal-efficencies. We do it in span 1, and then after
% including the dead time we compress to the wanted span.
% a) Crystal efficencies. A sinogram is generated from the crystal
% efficencies:
scanner_time_variant_ncf_3d = createSinogram3dFromDetectorsEfficency(used_xtal_efficiencies, structSizeSino3d_span1, 0);
% the ncf is 1/efficency:
nonzeros = scanner_time_variant_ncf_3d ~= 0;
scanner_time_variant_ncf_3d(nonzeros) = 1./ scanner_time_variant_ncf_3d(nonzeros);

% 6) Generate acquisition_dependant_ncf_3d (sinogram)
% The acquisition dependent factors can be only separated for span-1.
% Because infacts, it converts into a only one factor per detector:
% bin(d1,d2) = ef_d1*dt_d1*ef_d2*dt_d2.
% And then i need to do the axial compression.

% a) Get dead-time:
if ~isempty(singles_rates_per_bucket)
    %% GET PARALYZING AND NON PARALYZING DEAD TIME
    parDeadTime = componentFactors{5};
    nonParDeadTime = componentFactors{6};
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
else
    acquisition_dependant_ncf_3d = ones(size(scanner_time_variant_ncf_3d));
end

% 7) Overall factor:
% First compress scanner_time_variant_ncf_3d*acquisition_dependant_ncf_3d
% to the span we want to use:
[mixedTimeInvAcqDep_compressed, structSizeSino3d_span11] = convertSinogramToSpan(scanner_time_variant_ncf_3d.*acquisition_dependant_ncf_3d, structSizeSino3d_span1, structSizeSino3d.span);
% Just to have an approximation:
[scanner_time_variant_ncf_3d, structSizeSino3d_span11] = convertSinogramToSpan(scanner_time_variant_ncf_3d, structSizeSino3d_span1, structSizeSino3d.span);
[acquisition_dependant_ncf_3d, structSizeSino3d_span11] = convertSinogramToSpan(acquisition_dependant_ncf_3d, structSizeSino3d_span1, structSizeSino3d.span);
% Crystal dependent (crystal efficienies and crystal interference factors):
crystal_dependant_ncf_3d =  scanner_time_variant_ncf_3d;
for i = 1 : sum(structSizeSino3d.sinogramsPerSegment)
    crystal_dependant_ncf_3d(:,:,i) =  (1./crystalInterfFactor) .* crystal_dependant_ncf_3d(:,:,i) ./ axialFactors_xtal_dependent(i);
end

% Normalize to the number of sino mashed (because that is taken into
% account in the axial factors):
% Generate scanner time invariant:
for i = 1 : sum(structSizeSino3d.sinogramsPerSegment)
    % Normalize
    mixedTimeInvAcqDep_compressed(:,:,i) = mixedTimeInvAcqDep_compressed(:,:,i) ./ structSizeSino3d.numSinosMashed(i);
    scanner_time_variant_ncf_3d(:,:,i) = scanner_time_variant_ncf_3d(:,:,i) ./ structSizeSino3d.numSinosMashed(i);
    acquisition_dependant_ncf_3d(:,:,i) = acquisition_dependant_ncf_3d(:,:,i) ./ structSizeSino3d.numSinosMashed(i);
end
% Add gaps to time invariant:
gaps = scanner_time_variant_ncf_3d ~= 0;
scanner_time_invariant_ncf_3d = scanner_time_invariant_ncf_3d.*gaps;


overall_ncf_3d = scanner_time_invariant_ncf_3d .* mixedTimeInvAcqDep_compressed;

used_axial_factors = axialFactors;


