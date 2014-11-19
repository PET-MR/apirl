%% TESTBENCH
% Autor: Martín Belzunce
% Última Fecha de Modificaci�n: 26/11/08
%
% Grafico los filtros utilizados en backprojection
clear all
close all
addpath('/sources/MATLAB/WorkingCopy/ImageRecon')
%% INICIALIZACION DE FILTRO
% Inicializo los valores del filtro a partir del filtro seleccionado y la
% frecuncia de corte pasada
CantR = 167;
Fc = 1.0;
order = 2^nextpow2(2*CantR);           % El orden del vector filtro lo hago potencia de dos para que le pueda aplicar la fft
H = 2*( 0:(order/2) )./order;       % Inicializo los valores de la Respueta del Filtro como un Filtro Rampa, ya que los dem�s
                                    % son una modulaci�n de la rampa
w = 2*pi*(0:size(H,2)-1)/order;     % Valores de frecuencia

h = figure;
% Filtro Rampa
% No tengo que modificar la respuesta del filtro, simplemente pongo
% en cero los valores que excedan la Freucnecia de corte
H(w>pi*Fc) = 0;
subplot(2,3,1)
plot(w,H, 'LineWidth', 4);
title('Filtro Rampa','FontSize',20,'FontWeight','Bold');
ylabel('H(\omega)','FontSize',18,'FontWeight','Bold');
xlabel('\omega','FontSize',18,'FontWeight','Bold');


% Filtro Shepp-Logan
H = 2*( 0:(order/2) )./order;       % Inicializo los valores de la Respueta del Filtro como un Filtro Rampa, ya que los dem�s
                                    % son una modulaci�n de la rampa
H(2:end) = H(2:end).* (sin(w(2:end)/(2*Fc))./(w(2:end)/(2*Fc))); %H = H .* 2 .* Fc ./ pi .* sin(pi.*H./(2*Fc)); 
subplot(2,3,2)
plot(w,H, 'LineWidth', 4);
title('Filtro Shepp-Logan','FontSize',20,'FontWeight','Bold');
ylabel('H(\omega)','FontSize',18,'FontWeight','Bold');
xlabel('\omega','FontSize',18,'FontWeight','Bold');

% Filtro Coseno
H = 2*( 0:(order/2) )./order;       % Inicializo los valores de la Respueta del Filtro como un Filtro Rampa, ya que los dem�s
                                    % son una modulaci�n de la rampa
H(2:end) = H(2:end) .* cos(w(2:end)/(2*Fc)); %H = H .* cos(w/(2*Fc));
subplot(2,3,3)
plot(w,H, 'LineWidth', 4);
title('Filtro Coseno','FontSize',20,'FontWeight','Bold');
ylabel('H(\omega)','FontSize',18,'FontWeight','Bold');
xlabel('\omega','FontSize',18,'FontWeight','Bold');

% Filtro Hamming
H = 2*( 0:(order/2) )./order;       % Inicializo los valores de la Respueta del Filtro como un Filtro Rampa, ya que los dem�s
                                    % son una modulaci�n de la rampa
H(2:end) = H(2:end) .* (.54 + .46 * cos(w(2:end)/Fc)); %H = H .* (.54 + .46 * cos(w./Fc));
subplot(2,3,4)
plot(w,H, 'LineWidth', 4);
title('Filtro Hamming','FontSize',20,'FontWeight','Bold');
ylabel('H(\omega)','FontSize',18,'FontWeight','Bold');
xlabel('\omega','FontSize',18,'FontWeight','Bold');

% Filtro Hann
H = 2*( 0:(order/2) )./order;       % Inicializo los valores de la Respueta del Filtro como un Filtro Rampa, ya que los dem�s
                                    % son una modulaci�n de la rampa
H(2:end) = H(2:end) .*(1+cos(w(2:end)./Fc))./ 2; %H = 0.5*H.*(1+cos(w./Fc));
subplot(2,3,5)
plot(w,H, 'LineWidth', 4);
title('Filtro Hann','FontSize',20,'FontWeight','Bold');
ylabel('H(\omega)','FontSize',18,'FontWeight','Bold');
xlabel('\omega','FontSize',18,'FontWeight','Bold');

set(gcf, 'Position', [0 0 1400 700]);
set(gcf, 'InvertHardcopy', 'off')
outputGraphsPath = '/workspaces/Martin/Doctorado/Tesis/Tesis Martín Belzunce/docusTesis/Figuras/Capitulo3/backprojection/';
graphicFilename = sprintf('filtros');
set(gcf,'PaperPositionMode','auto');    % Para que lo guarde en el tamaño modificado.
saveas(gcf, [outputGraphsPath graphicFilename], 'fig');
frame = getframe(gcf);
imwrite(frame.cdata, [outputGraphsPath graphicFilename '.png']);
saveas(gca, [outputGraphsPath graphicFilename], 'epsc');
addpath('/sources/MATLAB/WorkingCopy/export_fig/');
export_fig '/workspaces/Martin/Doctorado/Tesis/Tesis Martín Belzunce/docusTesis/Figuras/Capitulo3/backprojection/filtros.pdf'

export_fig '/workspaces/Martin/Doctorado/Tesis/Tesis Martín Belzunce/docusTesis/Figuras/Capitulo3/backprojection/filtros.png'