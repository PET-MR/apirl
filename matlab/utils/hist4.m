%  *********************************************************************
%  Proyecto AR-PET. Comisión Nacional de Energía Atómica.
%  Autor: Martín Belzunce. UTN-FRBA.
%  Fecha de Creación: 08/07/2009
%  *********************************************************************

function Volumen = hist4(Puntos, CellCentros)

CentrosX = CellCentros{1};
CentrosY = CellCentros{2};
CentrosZ = CellCentros{3};

CellCentros = {CellCentros{1}(:)' CellCentros{2}(:)' CellCentros{3}(:)'};
nbins = [length(CellCentros{1}) length(CellCentros{2}) length(CellCentros{3})];
[nrows,ncols] = size(Puntos);

% Bin each observation in the x-direction, and in the y-direction.
bin = zeros(nrows,2);
% La matriz Puntos tiene 3 columnas con las coordenadas [X,Y,Z]

for i = 1:3
        minx = min(Puntos(:,i));
        maxx = max(Puntos(:,i));
            
        % If the bin centers were given, compute their edges and width
            c = CellCentros{i};
            dc = diff(c);
            edges{i} = [c(1) c] + [-dc(1) dc dc(end)]/2;
            binwidth{i} = diff(edges{i});
            % Make histc mimic hist behavior:  everything < ctrs(1) gets
            % counted in first bin, everything > ctrs(end) gets counted in
            % last bin.  ctrs, edges, and binwidth do not reflect that, but
            % histcEdges does.
            histcEdges = [-Inf edges{i}(2:end-1) Inf];
                  
        % Get the 1D bin numbers for this column of x.  Make sure +Inf
        % goes into the nth bin, not the (n+1)th.
        [dum,bin(:,i)] = histc(Puntos(:,i),histcEdges,1);
        bin(:,i) = min(bin(:,i),nbins(i));
end

% Combine the two vectors of 1D bin counts into a grid of 2D bin
% counts.
Volumen = accumarray(bin(all(bin>0,2),:),1,nbins);