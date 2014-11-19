%  *********************************************************************
%  Proyecto AR-PET. Comisión Nacional de Energía Atómica.
%  Autor: Martín Belzunce. UTN-FRBA.
%  Fecha de Creación: 24/02/2013
%  *********************************************************************
%
%  Grafico los priors.
clear all
close all
r = 1 : 256; % Diferencia entre píxeles.
delta_Geman = 100;
delta_Herbert = 100;
delta_Green = 100;
%% Funciones de energía
h = figure;
plot(r,1/45.*r.^2./(1+(r./delta_Geman).^2), r, delta_Herbert.*log(1+(r./delta_Herbert).^2), ...
    r, delta_Green.*log(cosh(r./delta_Green)), 'LineWidth',3);
h2=legend('Geman-McClure', 'Hebert-Leahy', 'Green','Location','SouthEast');
set(h2, 'FontSize',16);
xlabel('r', 'FontSize',16);
ylabel('V(r)', 'FontSize',16);
set(h, 'Position', [100 100 800 600]);



outputGraphsPath = '/workspaces/Martin/Doctorado/Tesis/Tesis Martín Belzunce/docusTesis/Figuras/Capitulo3/';
graphicFilename = sprintf('priorsMAP');
set(gcf,'PaperPositionMode','auto');    % Para que lo guarde en el tamaño modificado.
saveas(gcf, [outputGraphsPath graphicFilename], 'fig');
frame = getframe(gcf);
imwrite(frame.cdata, [outputGraphsPath graphicFilename '.png']);
saveas(gca, [outputGraphsPath graphicFilename], 'epsc');
saveas(gca, [outputGraphsPath graphicFilename], 'tif');
