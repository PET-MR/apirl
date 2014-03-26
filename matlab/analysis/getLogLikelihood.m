%  *********************************************************************
%  Proyecto TGS. Comisión Nacional de Energía Atómica.
%  Autor: Martín Belzunce. UTN-FRBA.
%  Fecha de Creación: 14/12/2011
%  *********************************************************************

% function [iteracion, logLikelihood] = getLogLikelihood(logFilename)

% Función que lee un archivo .log, y devuelve dos vectores, uno con los
% índices de iteración y el otro con log likelihood de la reconstrucción.

function [iteracion, logLikelihood] = getLogLikelihood(logFilename)

iteracion = [];
logLikelihood = [];
titleLikelihood = 'Likelihood por Iteración';

% Abro el archivo de log, y busco la línea de título del likelihhod, en la
% línea siguientes, tengo los valores separados por ','.
fid = fopen(logFilename);
if(fid == -1)
    disp('Error al abrir el archivo de log.')
    return
end
fullText = textscan(fid,'%s', 'Delimiter', '\n');

% Recorro las líneas, a ver en cual está el título:
lineLikelihood = -1;
for i = 1 : numel(fullText{1})
    if ~isempty(strfind(fullText{1}{i}, titleLikelihood))
        % Si encontró el título, los valores están en la línea siguientes:
        lineLikelihood = i + 1;
        break;
    end
end

if lineLikelihood == -1
    % No se encontré el título:
    disp(sprintf('No se encontró el título %s en el archivo de log.'));
    return;
end

logLikelihood = textscan(fullText{1}{lineLikelihood}, '%f', 'Delimiter', ',');
logLikelihood = logLikelihood{1};
iteracion = 1 : numel(logLikelihood);

