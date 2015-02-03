%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 15/01/2015
%  *********************************************************************
%  function [componentFactors, componentLabels]  = readmMrComponentBasedNormalization(filenameRawData, visualize)
% 
%  This functions read the component based normalization file of the mMr.
%  Receives two parameters:
%   -filenameRawData: the name of the file with the raw data(*.n).
%   -visualize: flag to indicate if a visualization of each factor is
%   desired.
%
%  The ouput is two cell arrays:
%   -componentFactors: each element of this cell array is a matrix or
%   vector with one of the component of the normalization.
%   -componentLabels: cell array of the same size of componentFactors,
%   where each of the elements is the name of the component. 
% 
%  The size of each component matrix are hardcoded for the mMr scanner and
%  are taken from the following specification of the interfile header:
%
% %NORMALIZATION COMPONENTS DESCRIPTION:=
% 
% %number of normalization components:=8
% %normalization component [1]:=geometric effects
% %normalization component [2]:=crystal interference
% %normalization component [3]:=crystal efficiencies
% %normalization component [4]:=axial effects
% %normalization component [5]:=paralyzing ring DT parameters
% %normalization component [6]:=non-paralyzing ring DT parameters
% %normalization component [7]:=TX crystal DT parameter
% %normalization component [8]:=additional axial effects
% 
% data offset in bytes [1]:=0
% data offset in bytes [2]:=174752
% data offset in bytes [3]:=187136
% data offset in bytes [4]:=316160
% data offset in bytes [5]:=319508
% data offset in bytes [6]:=319764
% data offset in bytes [7]:=320020
% data offset in bytes [8]:=320056
% 
% number of dimensions [1]:=2
% number of dimensions [2]:=2
% number of dimensions [3]:=2
% number of dimensions [4]:=1
% number of dimensions [5]:=1
% number of dimensions [6]:=1
% number of dimensions [7]:=1
% number of dimensions [8]:=1
% 
% %matrix size [1]:={344,127}
% %matrix size [2]:={9,344}
% %matrix size [3]:={504,64}
% %matrix size [4]:={837}
% %matrix size [5]:={64}
% %matrix size [6]:={64}
% %matrix size [7]:={9}
% %matrix size [8]:={837}
% 
% %matrix axis label [1]:={sinogram projection bins,sinogram planes}
% %matrix axis label [2]:={crystal number,sinogram projection bins}
% %matrix axis label [3]:={crystal number,ring number}
% %matrix axis label [4]:={plane number}
% %matrix axis label [5]:={ring number}
% %matrix axis label [6]:={ring number}
% %matrix axis label [7]:={crystal number}
% %matrix axis label [8]:={plane number}
% 
% %matrix axis unit [1]:={mm/pixel,mm/pixel}
% %matrix axis unit [2]:={mm/pixel,mm/pixel}
% %matrix axis unit [3]:={mm/pixel,mm/pixel}
% %matrix axis unit [4]:={mm/pixel}
% %matrix axis unit [5]:={mm/pixel}
% %matrix axis unit [6]:={mm/pixel}
% %matrix axis unit [7]:={mm/pixel}
% %matrix axis unit [8]:={mm/pixel}
% 
% %scale factor [1]:={2.0445,2.03125}
% %scale factor [2]:={2.0445,2.0445}
% %scale factor [3]:={2.0445,4.0625}
% %scale factor [4]:={2.03125}
% %scale factor [5]:={4.0625}
% %scale factor [6]:={4.0625}
% %scale factor [7]:={2.03125}
% %scale factor [8]:={2.03125}
% 
% %axial compression:=11
% %maximum ring difference:=60
% number of rings:=64
% number of energy windows:=1
% %energy window lower level (keV) [1]:=430
% %energy window upper level (keV) [1]:=610

function [componentFactors, componentLabels]  = readmMrComponentBasedNormalization(filenameRawData, visualize)

% Open file:
fid = fopen(filenameRawData, 'r');
if fid == -1
    ferror(fid);
end
% Read all the files from the specification, the sizes are fixed but it
% would be better to take the size fro the interfile header:
geomEffects = fread(fid,[344 127], 'single=>single');
crystalInterf = fread(fid,[9 344], 'single=>single');
crystalEff = fread(fid,[504 64], 'single=>single');
axialEffects = fread(fid,[837], 'single=>single');
parRingDT = fread(fid,[64], 'single=>single'); % Paralyzing Death Time per Ring.
nonParRingDT = fread(fid,[64], 'single=>single');  % Non-Paralyzing Death Time per Ring.
txCrystalDT = fread(fid,[9], 'single=>single');
otherAxialEffects = fread(fid,[837], 'single=>single');
fclose(fid);

componentFactors = {geomEffects, crystalInterf, crystalEff, axialEffects, parRingDT, nonParRingDT, txCrystalDT, otherAxialEffects};
componentLabels = {'Geometric Effects','Crystal Interference','Crystal Efficencies','Axial Effects','Paralyzing Dead Time','non-Paralyzing Dead Time','Transmission Dead Time','Other Axial Effects'};

% If the visualization flag is activated, show the factors:
if visualize==1
    % Plot the normalization parameters:
    h = figure;
    set(gcf, 'Position', [0 0 1600 1200]);
    showImageWithColorbar(geomEffects, colormap);
    title(componentLabels{1});
    
    h = figure;
    set(gcf, 'Position', [0 0 1600 1200]);
    subplot(1,2,1);
    imshow(crystalInterf'./max(max(crystalInterf)));
    title(componentLabels{2});
    subplot(1,2,2);
    imshow(crystalEff./max(max(crystalEff)));
    title(componentLabels{3});
    h = figure;
    set(gcf, 'Position', [0 0 1600 1200]);
    plot(crystalEff(:,1:10));
    title('Crystal Efficencies for the First 10 Rings');
    for i = 1 : 10
        labels{i} = sprintf('Ring %d', i);
    end
    text(200, 200, sprintf('StdDev: %f %%', std(crystalEff(:))));
    legend(labels);
    
    h = figure;
    set(gcf, 'Position', [0 0 1600 1200]);
    set(gcf, 'Position', [0 0 1600 1200]);
    subplot(1,2,1);
    plot(axialEffects);
    title(componentLabels{4});
    subplot(1,2,2);
    plot(otherAxialEffects);
    title(componentLabels{8});

    h = figure;
    set(gcf, 'Position', [0 0 1600 1200]);
    subplot(1,3,1);
    plot(parRingDT);
    title(componentLabels{5});
    subplot(1,3,2);
    plot(nonParRingDT);
    title(componentLabels{6});
    subplot(1,3,3);
    plot(txCrystalDT);
    title(componentLabels{7});
end