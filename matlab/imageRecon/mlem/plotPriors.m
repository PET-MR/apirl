%  *********************************************************************
%  Proyecto AR-PET. Comisión Nacional de Energía Atómica.
%  Autor: Martín Belzunce. UTN-FRBA.
%  Fecha de Creación: 24/02/2013
%  *********************************************************************
%
%  Grafico los priors.
clear all
close all
r = -256: 1 : 256; % Diferencia entre píxeles.
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


%% LANGE
h = figure;
r = -1: 0.01 : 1; % Diferencia entre píxeles.
delta = [1e-3*max(abs(r)) 5e-3*max(abs(r)) 1e-2*max(abs(r)) 5e-2*max(abs(r)) 1e-1*max(abs(r)) 2e-1*max(abs(r)) max(abs(r))];
[DELTA, R] = meshgrid(delta,r);
% Equalization factor:
scale_factor = (max(abs(r))./2./delta(1)-log(1+max(abs(r))./2./delta(1)))./(max(abs(r))./2./DELTA-log(1+max(abs(r))./2./DELTA));
lange = scale_factor.*(abs(R)./DELTA-log(1+abs(R)./DELTA));
plot(R,lange, 'LineWidth', 2);
for i = 1 : numel(delta)
    labels{i} = sprintf('\\delta=%d', delta(i));
end
legend(labels, 'Location', 'NorthWest');
title('Lange/Fair');
ylabel('\psi(t)');
xlabel('t');

% Derivada
h = figure;
scale_factor = (max(abs(R))/2./(max(abs(R))./2+delta(1)))./(max(abs(R))/2./(max(abs(R))./2+delta));
dlange = scale_factor.*(abs(R)./(abs(R)+DELTA));
plot(R,dlange, 'LineWidth', 2);
for i = 1 : numel(delta)
    labels{i} = sprintf('\\delta=%d', delta(i));
end
legend(labels, 'Location', 'NorthWest');
title('Lange/Fair');
ylabel('\psi(t)');
xlabel('t');

%%
h = figure;
delta = [1, 20:20:100, 100:50:300];
delta = 1 : 2 :20;
[DELTA, R] = meshgrid(delta,r);
huber = zeros(size(DELTA));
huber(DELTA>=abs(R)) = R(DELTA>=abs(R)).^2/2;
huber(DELTA<abs(R)) = DELTA(DELTA<abs(R)).*R(DELTA<abs(R))-DELTA(DELTA<abs(R)).^2/2;
plot(R,huber, 'LineWidth', 2);
for i = 1 : numel(delta)
    labels{i} = sprintf('\\delta=%d', delta(i));
end
legend(labels, 'Location', 'NorthWest');
ylabel('\psi(t)');
xlabel('t');

h = figure;
[DELTA, R] = meshgrid(delta,r);
dhuber = zeros(size(DELTA));
dhuber(DELTA>=R) = R(DELTA>=R);
dhuber(DELTA<R) = DELTA(DELTA<R);
plot(R,dhuber, 'LineWidth', 2);
for i = 1 : numel(delta)
    labels{i} = sprintf('\\delta=%d', delta(i));
end
legend(labels, 'Location', 'NorthWest');
ylabel('\psi(t)');
xlabel('t');