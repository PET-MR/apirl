%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 23/02/2016
%  *********************************************************************
%  Function that makes the fan sum from
function [crystalCounts] = FansumPerCrystalMmr(sinogram, structSizeSino3d, crystalIndex, method)

% By default method 1:
if nargin == 2 
    method = 1;
end
% Crystals in the mmr scanner:
numRings = 64;
numDetectorsPerRing = 504;


if method == 1 
    % Maps qith the id of each crystal in the sinogram:
    [mapaDet1Ids, mapaDet2Ids] = createMmrDetectorsIdInSinogram3d();

    % For each crystal, get all the sinogram bins where its involved and sum
    % the counts:
    crystalAsDet1 = mapaDet1Ids == crystalIndex;
    crystalAsDet2 = mapaDet2Ids == crystalIndex;
    crystalCounts = sum(sinogram(crystalAsDet1|crystalAsDet2));
elseif method == 2
    disp('Method 2 not implemented for individual crystals.');
elseif method == 3
    % I go for each bin and sum in the respective crystal the counts:
    % Maps qith the id of each crystal in the sinogram:
    [mapaDet1Ids, mapaDet2Ids] = createMmrDetectorsIdInSinogram3d();
    crystalCounts = 0;
    for i = 1 : size(sinogram,1)
        for j = 1 : size(sinogram,2)
            for k = 1 : size(sinogram,3)
                if(mapaDet1Ids(i,j,k)==crystalIndex)||(mapaDet2Ids(i,j,k)==crystalIndex)
                    crystalCounts = crystalCounts + sinogram(i,j,k);
                end
            end
        end
    end
end

