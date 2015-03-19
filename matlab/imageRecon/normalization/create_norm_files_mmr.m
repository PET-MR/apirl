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

function [overall_ncf_3d, scanner_time_invariant_ncf_3d, scanner_time_variant_ncf_3d, acquisition_dependant_ncf_3d, used_xtal_efficiencies, used_deadtimefactors, used_axial_factors] = ...
   create_norm_files_mmr(cbn_filename, my_axial_factors, my_selection_of_xtal_efficiencies, my_choice_of_deadtimefactors, singles_rates_per_bucket, span_choice)


used_xtal_efficiencies = [];
used_deadtimefactors = [];
used_axial_factors = [];

% 1) Read the .n files and get each component in a cell array:
[componentFactors, componentLabels]  = readmMrComponentBasedNormalization(cbn_filename, 0);

% 2) Size of the mmr's sinograms. The 3d parameters are generated from the
% selected span:
% Size of mMr Sinogram's
numTheta = 252; numR = 344; numRings = 64; maxAbsRingDiff = 60; rFov_mm = 594/2; zFov_mm = 258;
structSizeSino3d = getSizeSino3dFromSpan(numR, numTheta, numRings, rFov_mm, zFov_mm, span_choice, maxAbsRingDiff);
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
% Generate the sinograms:
overall_ncf_3d = zeros(numR, numTheta, numSinograms, 'single');
scanner_time_invariant_ncf_3d = zeros(numR, numTheta, numSinograms, 'single');
scanner_time_variant_ncf_3d = zeros(numR, numTheta, numSinograms, 'single');
acquisition_dependant_ncf_3d = zeros(numR, numTheta, numSinograms, 'single');

% 3) Selection of dead-time parameters and crystal efficencies:
if isempty(my_selection_of_xtal_efficiencies)
    % Use the crystal efficencies of the .n file:
    used_xtal_efficiencies = componentFactors{3};
else
    % Use the crystal efficencies received in the parameter file:
    used_xtal_efficiencies = my_selection_of_xtal_efficiencies;
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
if span_choice == 11
    axialFactors = structSizeSino3d.numSinosMashed'.* (componentFactors{4}.*componentFactors{8});
else
    axialFactors = structSizeSino3d.numSinosMashed';
end

% Generate scanner time invariant:
for i = 1 : sum(structSizeSino3d.sinogramsPerSegment)
    % First the geomeitric, crystal interference factors:
    scanner_time_invariant_ncf_3d(:,:,i) = 1./(geometricFactor .* crystalInterfFactor);
    % Axial factor:
    scanner_time_invariant_ncf_3d(:,:,i) = scanner_time_invariant_ncf_3d(:,:,i) .* (1./axialFactors(i));
end

% 5) We generate the scan_dependent_ncf_3d, that is compound by the
% dead-time and crystal-efficencies.
% a) Crystal efficencies. A sinogram is generated from the crystal
% efficencies:
scanner_time_variant_ncf_3d = createSinogram3dFromDetectorsEfficency(used_xtal_efficiencies, structSizeSino3d, 0);
% the ncf is 1/efficency:
nonzeros = scanner_time_variant_ncf_3d ~= 0;
scanner_time_variant_ncf_3d(nonzeros) = 1./ scanner_time_variant_ncf_3d(nonzeros);

% 6) Generate acquisition_dependant_ncf_3d (sinogram)
% a) Get dead-time:
if ~isempty(singles_rates_per_bucket)
    % Get the single rate per ring:
    singles_rates_per_ring = singles_rates_per_bucket / (numberOfAxialBlocksPerBucket*numberOfAxialCrystalsPerBlock);
    % Compute dead time factors, is equivalent to an efficency factor.
    % Thus, a factor for each crystal unit is computed, and then both
    % factors are multiplied.
    % Detector Ids:
    [mapDet1Ids, mapDet2Ids] = createMmrDetectorsIdInSinogram();
    % BucketId:
    mapBucket1Ids = ceil(mapDet1Ids/(numberOfTransverseBlocksPerBucket*numberOfTransverseCrystalsPerBlock));
    mapBucket2Ids = ceil(mapDet2Ids/(numberOfTransverseBlocksPerBucket*numberOfTransverseCrystalsPerBlock));
    % Now we start going through each possible sinogram, then get the rings of
    % each sinogram and get the crystal efficency for that ring in det1 and the
    % crystal effiencies for the sencdo ring with det2. When there is axial
    % compression the efficency is computed as an average of each of them. For
    % example an sinogram compressed from two different axial position:
    % det1(ring1_a)*det2(ring2_a)+det1(ring1_b)*det2(ring2_b)
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
                        % Get the singles rate for each ring:
                        axialBucket1 = ceil(z1_aux / (numberOfAxialBlocksPerBucket*numberOfAxialCrystalsPerBlock));
                        axialBucket2 = ceil(z2 / (numberOfAxialBlocksPerBucket*numberOfAxialCrystalsPerBlock));
                        bucketsId1 = mapBucket1Ids + (axialBucket1-1)*numberOfBucketsInRing;
                        bucketsId2 = mapBucket2Ids + (axialBucket2-1)*numberOfBucketsInRing;
                        acquisition_dependant_ncf_3d(:,:,indiceSino) = acquisition_dependant_ncf_3d(:,:,indiceSino) + singles_rates_per_bucket(bucketsId1) .* singles_rates_per_bucket(bucketsId2);
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

% 6) Overall factor:
overall_ncf_3d = scanner_time_invariant_ncf_3d .* scanner_time_variant_ncf_3d;



