%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 09/02/2015
%  *********************************************************************
%  function [overall_ncf_2d, scan_independent_ncf_2d, scan_dependent_ncf_2d, used_xtal_efficiencies, used_deadtimefactors] = ...
%       create_norm_files_mmr(cbn_filename, my_selection_of_xtal_efficiencies, my_choice_of_deadtimefactors, singles_rates_per_bucket, span_choice)
% 
%  This functions read the component based normalization file of the mMr
%  and creates the normalization factors for 2d sinograms. Optionally it can use externals crystal efficencies.
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
%   -numRings: number of rings, it will generate one direct sinogram per ring. 
%
%  The function returns de following outputs:
%   -overall_ncf_2d: overall normalization correction factors. Its the
%       multiplication of the scan_independent_ncf_2d with the
%       scan_dependent_ncf_2d.
%   -scan_independent_ncf_2d: normalization correction factors without scan
%       dependant factors (dead time and crystal efficencies).
%   -scan_dependent_ncf_2d: normalization correction factors for scan
%       dependant factors (dead-time and crystal efficencies).
%   -used_xtal_efficiencies: crystal effincecies factors used in the
%       overall_ncf_2d. 
%   -used_deadtimefactors: dead time contstants used for compute the dead
%   time factors.
% 
%  The size of each component matrix are hardcoded for the mMr scanner and
%  are

function [overall_ncf_2d, scanner_time_invariant_ncf_2d, scanner_time_variant_ncf_2d, acquisition_dependant_ncf_2d, crystal_dependant_ncf_2d, used_xtal_efficiencies, used_deadtimefactors, used_axial_factors, structSizeSino2d] = ...
   create_norm_files_mmr(cbn_filename, my_selection_of_xtal_efficiencies, my_choice_of_deadtimefactors, singles_rates_per_bucket, numRings)


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

% 2) Size of the mmr's sinograms. The 2d parameters are generated from the
% selected span:
% Size of mMr Sinogram's
numTheta = 252; numR = 344; 
structSizeSino2d = getSizeSino2dStruct(numR, numTheta, numRings, 596, 260);

% Total number of singorams per 2d sinogram:
numSinograms = sum(structSizeSino2d.numZ);
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
overall_ncf_2d = zeros(numR, numTheta, numSinograms, 'single');
scanner_time_invariant_ncf_2d = zeros(numR, numTheta, numSinograms, 'single');
scanner_time_variant_ncf_2d = zeros(numR, numTheta, numSinograms, 'single');
acquisition_dependant_ncf_2d = zeros(numR, numTheta, numSinograms, 'single');

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

% 4) We generate the scan_independent_ncf_2d, that is compound by the
% geometricFactor, the crystal interference and the axial effects:
% a) Geometric Factor. The geomtric factor is one projection profile per
% plane. But it's the same for all planes, so I just use one of them.
geometricFactor = repmat(single(componentFactors{1}(:,1)), 1, structSizeSino2d.numTheta);
% b) Crystal interference, its a pattern that is repeated peridoically:
crystalInterfFactor = single(componentFactors{2});
crystalInterfFactor = repmat(crystalInterfFactor', 1, structSizeSino2d.numTheta/size(crystalInterfFactor,1));

% Generate scanner time invariant:
for i = 1 : sum(structSizeSino2d.numZ)
    % First the geomeitric, crystal interference factors:
    scanner_time_invariant_ncf_2d(:,:,i) = 1./(geometricFactor .* crystalInterfFactor);
end

% 5) We generate the scan_dependent_ncf_2d, that is compound by the
% dead-time and crystal-efficencies. We do it in span 1, and then after
% including the dead time we compress to the wanted span.
% a) Crystal efficencies. A sinogram is generated from the crystal
% efficencies:
for i = 1 : sum(structSizeSino2d.numZ)
    scanner_time_variant_ncf_2d(:,:,i) = createSinogram2dFromDetectorsEfficency(used_xtal_efficiencies(:,i), structSizeSino2d, 2, 0);
end
% the ncf is 1/efficency:
nonzeros = scanner_time_variant_ncf_2d ~= 0;
scanner_time_variant_ncf_2d(nonzeros) = 1./ scanner_time_variant_ncf_2d(nonzeros);

% 6) Generate acquisition_dependant_ncf_2d (sinogram)
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
    % Get the dead time for each detector and then multipliy to get the one
    % for the bin.
    deadTimeFactorsD1 = (1 + parDeadTime(1)*singles_rates_per_bucket(map2dBucket1Ids)/2 - nonParDeadTime(1).*singles_rates_per_bucket(map2dBucket1Ids).*singles_rates_per_bucket(map2dBucket1Ids)/4);
    deadTimeFactorsD2 = (1 + parDeadTime(1)*singles_rates_per_bucket(map2dBucket2Ids)/2 - nonParDeadTime(1).*singles_rates_per_bucket(map2dBucket2Ids).*singles_rates_per_bucket(map2dBucket2Ids)/4);
    acquisition_dependant_ncf_2d = 1./(deadTimeFactorsD1 .* deadTimeFactorsD2);
else
    acquisition_dependant_ncf_2d = ones(size(scanner_time_variant_ncf_2d));
end

% Crystal dependent (crystal efficienies and crystal interference factors):
crystal_dependant_ncf_2d =  (1./crystalInterfFactor) .* scanner_time_variant_ncf_2d;

% Add gaps to time invariant:
gaps = scanner_time_variant_ncf_2d ~= 0;
scanner_time_invariant_ncf_2d = scanner_time_invariant_ncf_2d.*gaps;


overall_ncf_2d = scanner_time_invariant_ncf_2d .* acquisition_dependant_ncf_2d .* scanner_time_variant_ncf_2d;


