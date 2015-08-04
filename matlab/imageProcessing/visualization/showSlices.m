%  *********************************************************************
%  Reconstruction Framework for Siemens Biograph mMR.  
%  Autor: Martín Belzunce. Kings College London.
%  Fecha de Creación: 07/07/2015
%  *********************************************************************
%  
%  function handle = showSlices(volume, pause_sec)
%
%  Shows an image slice by slice iwth a pause of pause_sec between them. It
%  return the handle to the figure.
function handle = showSlices(volume, pause_sec)

if nargin == 1
    pause_sec = 0.2;
end
handle = figure;
for i = 1 : size(volume,3)
    imshow(volume(:,:,i), [0 max(max(volume(:,:,i)))]);
    pause(pause_sec);
end