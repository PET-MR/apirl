%  *********************************************************************
%  Proyecto TGS. Comisión Nacional de Energía Atómica.
%  Autor: Martín Belzunce. UTN-FRBA.
%  Fecha de Creación: 14/12/2011
%  *********************************************************************

% Script que compara distintos proyectores para una reconstrucción en
% particular:

reconAtirlPath = '/sources/TGS/AnalisisResEspacial/branches/EnsayoMultiplesFuentesSpineBone/matlab/reconAtirl';

% Nombres de salida de las reconstrucciones a comparar:
reconNames = {'MLEM_SinoLowRes_SiddonWithAttenCorr_60x60', 'MLEM_SinoLowRes_CoR_withAttenCorr_5', ...
    'MLEM_SinoLowRes_CoR_withAttenCorr_10', 'MLEM_SinoLowRes_CoR_withAttenCorr_15', ...
    'MLEM_SinoLowRes_CoRwP_withAttenCorr_10_20', 'MLEM_SinoLowRes_CoRwP_withAttenCorr_20_30', ...
    'MLEM_SinoLowRes_CoRwP_withAttenCorr_30_45'};
% Nombres de salida para los títulos de los gráficos:
titles = {'Siddon w/ Atten Corr', 'CoR w/ Atten Corr 5p', 'CoR w/ Atten Corr 10p','CoR w/ Atten Corr 15p',...
    'CoRwP w/ Atten Corr 10,20p', 'CoRwP w/ Atten Corr 20,30p', 'CoRwP w/ Atten Corr 30,45p'}; 

colors = 'rgbcmyk';

h = figure;
for i = 1 : numel(reconNames)
   fileName = sprintf('%s/%s.log', reconAtirlPath, reconNames{i});
   [iteraciones, likelihood] = getLogLikelihood(fileName);
   plot(iteraciones(18:end-1), likelihood(18:end-1), colors(i));
   hold on
end
legend(titles, 'Location', 'SouthEast');
set(gca,'LineStyleOrder','-+|-*|-o');
title('Likelihood según proyector para Imagen de Múltiples Fuentes en Spine Bone');