%  *********************************************************************
%  Proyecto AR-PET. Comisión Nacional de Energía Atómica.
%  Autor: Martín Belzunce. UTN-FRBA.
%  Fecha de Creación: 23/08/2011
%  *********************************************************************
%  function structSimu = getSimuStruct(path, nameSimu, outputName,
%  numSplits, filesPerSplit)
% 
%  Función que genera una estructura con la información de una simulación a
%  procesar. Esta estructura se utiliza como parámetro en las funciones que
%  procesan una simulación, y contiene los siguientes campos:
%  
%  structSimu.path : Path del directorio donde se encuentra la simulación.
% 
%  structSimu.name: Nombre de la salida de la simulación (Primera
%  parte del nombre de los archivos de salida).
%
%  structSimu.digiName: Nombre de la salida del digitizer a procesar.
% 
%  structSimu.numSplits: Número de splits de la simulación.
%
%  structSimu.numFilesPerSplit: Número de archivos de salida por cada
%  split.Cuando se llega al límite de 1.9 GB gate, genera un nuevo archivo.
%  Recibe los valores de cada uno de esos campos como parámetro, y devuelve
%  la estructura en si.


function structSimu = getSimuStruct(path, nameSimu, outputName, numSplits, numFilesPerSplit)

structSimu = struct('path', path, 'name', nameSimu, 'digiName', outputName, 'numSplits', numSplits, 'numFilesPerSplit', numFilesPerSplit);
