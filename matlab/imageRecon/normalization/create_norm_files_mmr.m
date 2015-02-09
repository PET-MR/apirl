%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 09/02/2015
%  *********************************************************************
%  function [overall_ncf_3d, scan_independent_ncf_3d, scan_dependent_ncf_3d, used_xtal_efficiencies, used_deadtimefactors] = ...
%       create_norm_files_mmr(cbn_filename, my_selection_of_xtal_efficiencies, my_choice_of_deadtimefactors, span_choice)
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
%   want to use different dead time factors from the ones in the .n files.
%   If not just use an empty vector for this parameter [].
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
%   -used_deadtimefactors: dead time factors used in the overall_ncf_3d.
% 
%  The size of each component matrix are hardcoded for the mMr scanner and
%  are

function [overall_ncf_3d, scan_independent_ncf_3d, scan_dependent_ncf_3d, used_xtal_efficiencies, used_deadtimefactors] = ...
   create_norm_files_mmr(cbn_filename, my_selection_of_xtal_efficiencies, my_choice_of_deadtimefactors, span_choice)

% 1) Read the .n files and get each component in a cell array:
[componentFactors, componentLabels]  = readmMrComponentBasedNormalization(cbn_filename, 0);

% 2) Size of the mmr's sinograms. The 3d parameters are generated from the
% selected span:
% Size of mMr Sinogram's
numTheta = 252; numR = 344; numRings = 64; maxAbsRingDiff = 60; rFov_mm = 594/2; zFov_mm = 258;
structSizeSino3d = getSizeSino3dFromSpan(numTheta, numR, numRings, rFov_mm, zFov_mm, span_choice, maxAbsRingDiff);
% Total number of singorams per 3d sinogram:
numSinograms = sum(structSizeSino3d.sinogramsPerSegment);
% Generate the sinograms:
overall_ncf_3d = zeros(numTheta, numR, numSinograms, 'single');
scan_independent_ncf_3d = zeros(numTheta, numR, numSinograms, 'single');
scan_dependent_ncf_3d = zeros(numTheta, numR, numSinograms, 'single');

% 3) Selection of dead-time parameters and crystal efficencies:
if isempty(my_selection_of_xtal_efficiencies)
    % Use the crystal efficencies of the .n file:
    used_xtal_efficiencies = componentFactors{3};
else
    % Use the crystal efficencies received in the parameter file:
    used_xtal_efficiencies = my_selection_of_xtal_efficiencies;
end

if isempty(my_choice_of_deadtimefactors)
    % Use the deadtime of the .n file:
    used_deadtimefactors = componentFactors{6};
else
    % Use the deadtime efficencies received in the parameter file:
    used_deadtimefactors = my_choice_of_deadtimefactors;
end

% 4) We generate the scan_independent_ncf_3d, that is compound by the
% geometricFactor, the crystal interference and the axial effects:
% a) Geometric Factor. The geomtric factor is one projection profile per
% plane. But it's the same for all planes, so I just use one of them.
geometricFactor = repmat(single(componentFactors{1}(:,1))', structSizeSino3d.numTheta, 1);
% b) Crystal interference, its a pattern that is repeated peridoically:
crystalInterfFactor = single(componentFactors{2});
crystalInterfFactor = repmat(crystalInterfFactor,structSizeSino3d.numTheta/size(crystalInterfFactor,1),1);
% c) Axial factors:
axialFactors = 1./(componentFactors{4}.*componentFactors{8});

for i = 1 : sum(structSizeSino3d.sinogramsPerSegment)
    % First the geomeitric, crystal interference factors:
    scan_independent_ncf_3d(:,:,i) = 1./(geometricFactor .* crystalInterfFactor);
    % Axial factor:
    scan_independent_ncf_3d(:,:,i) = scan_independent_ncf_3d(:,:,i) .* (1./axialFactors(i));
end

% 5) We generate the scan_dependent_ncf_3d, that is compound by the
% dead-time and crystal-efficencies.
% a) Crystal efficencies. A sinogram is generated from the crystal
% efficencies:
scan_dependent_ncf_3d = createSinogram3dFromDetectorsEfficency(used_xtal_efficiencies, structSizeSino3d, 0);
% the ncf is 1/efficency:
nonzeros = scan_dependent_ncf_3d ~= 0;
scan_dependent_ncf_3d(nonzeros) = 1./ scan_dependent_ncf_3d(nonzeros);
 
% b) Get dead-time:
% Not implemented yet.

% 6) Overall factor:
overall_ncf_3d = scan_independent_ncf_3d .* scan_dependent_ncf_3d;



