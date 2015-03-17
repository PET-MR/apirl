%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 11/03/2015
%  *********************************************************************
%  This function shows the maximum intensity projection in the transverse,
%  sagital and coronal planes. It receives a volume.
function [mipTransverse, mipCoronal, mipSagital] = showMaximumIntensityProjections(volume, cmap, sizeFigure)

if nargin == 1
    sizeFigure = [50 50 1600 1200];
    cmap = hot;
elseif nargin == 2
    sizeFigure = [50 50 1600 1200];
end

% Get the Maximum Intensity ProjectionsL
% Transverse XY:
mipTransverse = max(volume,[],3);
% Coronal XZ:
mipCoronal = max(volume,[],2);
mipCoronal = permute(mipCoronal, [1 3 2]);
% Sagital YZ:
mipSagital = max(volume,[],1);
mipSagital = permute(mipSagital, [2 3 1]);

figure;
set(gcf, 'Position', sizeFigure);
set(gcf, 'Name', 'Maximum Intensity Projection');
% Transverse
subplot(1,3,1);
maxValue = max(max(mipTransverse));
imshow(mipTransverse, [0 maxValue]);
% Aplico el colormap
colormap(cmap);
% Cambio las leyendas a las unidades que me interesan:
hcb = colorbar;
%set(hcb, 'YTickLabelMode', 'manual');
set(hcb, 'FontWeight', 'bold');
title('Transverse');

% Coronal
subplot(1,3,2);
maxValue = max(max(mipCoronal));
imshow(mipCoronal, [0 maxValue]);
% Aplico el colormap
colormap(cmap);
% Cambio las leyendas a las unidades que me interesan:
hcb = colorbar;
%set(hcb, 'YTickLabelMode', 'manual');
set(hcb, 'FontWeight', 'bold');
title('Coronal');

% Coronal
subplot(1,3,3);
maxValue = max(max(mipSagital));
imshow(mipSagital, [0 maxValue]);
% Aplico el colormap
colormap(cmap);
% Cambio las leyendas a las unidades que me interesan:
hcb = colorbar;
%set(hcb, 'YTickLabelMode', 'manual');
set(hcb, 'FontWeight', 'bold');
title('Sagital');




