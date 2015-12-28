%  *********************************************************************
%  Proyecto TGS. Comisión Nacional de Energía Atómica.
%  Autor: Martín Belzunce. UTN-FRBA.
%  Fecha de Creación: 14/12/2011
%  *********************************************************************

% function [iteracion, logLikelihood] = getLogLikelihood(logFilename)

% Función que lee un archivo .log, y devuelve un cell array con los
% tiempos, primero tiempo por iteracion, luego por projector y luego por
% backprojector.

function [times_mseg] = getReconstructionTimes(logFilename)

iteracion = [];
logLikelihood = [];
titles = {'Tiempos de Reconstrucción por Iteración [mseg]', 'Tiempos de Forwardprojection por Iteración [mseg]',...
    'Tiempos de Backwardprojection por Iteración [mseg]'};

% Abro el archivo de log, y busco la línea de título del likelihhod, en la
% línea siguientes, tengo los valores separados por ','.
fid = fopen(logFilename);
if(fid == -1)
    disp('Error al abrir el archivo de log.')
    return
end
fullText = textscan(fid,'%s', 'Delimiter', '\n');

% Recorro las líneas, a ver en cual está el título:
lineTitle = -1;
for j = 1 : numel(titles)
    for i = 1 : numel(fullText{1})
        if ~isempty(strfind(fullText{1}{i}, titles{j}))
            % Si encontró el título, los valores están en la línea siguientes:
            lineTitle = i + 1;
            break;
        end
    end
    if lineTitle == -1
        % No se encontré el título:
        disp(sprintf('No se encontró el título %s en el archivo de log.', titles{j}));
        return;
    end
    aux = textscan(fullText{1}{lineTitle}, '%f', 'Delimiter', ',');
    times_mseg{j} = aux{1}; 
end

