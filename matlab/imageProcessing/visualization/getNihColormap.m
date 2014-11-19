%  *********************************************************************
%  Proyecto TGS. Comisión Nacional de Energía Atómica.
%  Autor: Martín Belzunce. UTN-FRBA.
%  Fecha de Creación: 04/04/2011
%  *********************************************************************

% Creo la paleta de colores NIH del AMIDE, saco los códigos de ahí.

function nihColormap = getNihColormap(min,max)

% El colormap lo hago de 256 tonos:
numColors = 256;
nihColormap = zeros(numColors,3);

% Si no me pasa los valores mínimos y máximos de la iamgen a visualizar,
% uso el estándar para imágenes del tipo double en matlab: 0-1.
if nargin == 0
    min = 0;
    max = 1;
end
% Creo un vector con los valores posibles de entrada a partir de mínimos y
% máximos:
inputVector = min : (max-min)/(numColors-1) : max;
inputVectorNorm = ((inputVector-min)./(max-min));
% La paleta de colores la creo en hsv primero:
hsvColormap = zeros(numColors,3);
% En esta conversión usa rgba, por lo que agrego un vector para el alpha:
alpha = zeros(numColors,1);
% La componente s la inicializo en 1:
hsvColormap(:,2) = 1;

% La componente v según:
% if (temp < 0.0)
%   hsv.v = 0.0;
% else if (temp > 0.2)
%   hsv.v = 1.0;
% else
%   hsv.v = 5*temp;
hsvColormap((inputVectorNorm<0.0),3) = 0;
hsvColormap((inputVectorNorm>0.2),3) = 1.0;
hsvColormap((inputVectorNorm>=0.0)&(inputVectorNorm<=0.2),3) = inputVectorNorm((inputVectorNorm>=0.0)&(inputVectorNorm<=0.2))/0.2;

% Ahora la componente h:
% if (temp > 1.0) {
%   hsv.h = 0.0;
%   rgba.a = 0xFF;
% } else if (temp < 0.0) {
%   hsv.h = 300.0;
%   rgba.a = 0;
% } else {
%   hsv.h = 300.0*(1.0-temp); 
%   rgba.a = 0xFF*temp;
% }
hsvColormap((inputVectorNorm>1.0),1) = 0;
alpha((inputVectorNorm>1.0)) = 1;
hsvColormap((inputVectorNorm<0.0),1) = 0;
alpha((inputVectorNorm<0.0),1) = 0;
hsvColormap((inputVectorNorm<1.0)&(inputVectorNorm>0.0),1) = 0.75*(1-inputVectorNorm((inputVectorNorm<1.0)&(inputVectorNorm>0.0)));
alpha((inputVectorNorm<1.0)&(inputVectorNorm>0.0),1) = 1 * inputVectorNorm((inputVectorNorm<1.0)&(inputVectorNorm>0.0));

% Convierto hsv a rgb:
nihColormap = hsv2rgb(hsvColormap);
% Le agrego la corrección de alpha (http://stackoverflow.com/questions/2049230/convert-rgba-color-to-rgb):
% Source => Target = (BGColor + Source) =
% Target.R = ((1 - Source.A) * Source.R) + (Source.A * BGColor.R)
% Target.G = ((1 - Source.A) * Source.G) + (Source.A * BGColor.G)
% Target.B = ((1 - Source.A) * Source.B) + (Source.A * BGColor.B)
% nihColormap(:,1) = ((1-alpha).*nihColormap(:,1)) + (alpha*0);
% nihColormap(:,2) = ((1-alpha).*nihColormap(:,2)) + (alpha*0);
% nihColormap(:,3) = ((1-alpha).*nihColormap(:,3)) + (alpha*0);
