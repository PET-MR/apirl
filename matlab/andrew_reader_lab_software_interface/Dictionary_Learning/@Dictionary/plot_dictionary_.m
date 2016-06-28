function H = plot_dictionary_(~,D,X,nb, options)

% plot_dictionnary - display a dictionary of images
%
%   H = plot_dictionnary(D,X,nb, options);
%
%   options.ndim gives dimensionality (1D/2D)
%   options.s gives number of channel
%
%   D is a (n,p) dictionary of p atoms of dimension n.
%
%   each atom can be a 2D image (ndim==2) of size (s,sqrt(n),sqrt(n))
%   or a 1D signal (ndim==1) of size n
%
%   Copyright (c) 2007 Gabriel Peyre

if nargin<2
    X = [];
end
if nargin<3
    nb = 10;
end
if length(nb)==1
    nb = [nb nb];
end


options.null = 0;
ndim = getoptions(options, 'ndim', 2);
s = getoptions(options, 's', 1);

n = size(D,1);
K = size(D,2);

w1 = sqrt(n/s);
pad = 2;

col = [1 1 1];

if ~isempty(X)
    d = sum( abs(X') );
    [tmp,I] = sort(d); I=I(end:-1:1);
else
%    I = randperm(K);
    I = round( linspace(1,K,prod(nb)) );
end

if ndim==1
    % display for 1D dictionary 
    clf;
    k = 0;
    hold on;
    for i=1:nb(1)
        for j=1:nb(2)
            k = k+1;
            if k<=length(I)
                subplot(nb(1), nb(2), k);
                plot(  D(:,I(k)) ); axis tight;
            end
        end
    end
    hold off;
    return;
end

% nb = min(15, floor(sqrt(K)));
H = repmat( reshape(col,[1 1 3]), [nb*(w1+pad) 1] );

vmax = max( max( abs( D(:,I(1:prod(nb))) ) ) );

normalization = getoptions(options, 'normalization', 'rescale');

k = 0;
for i=1:nb(1)
    for j=1:nb(2)
        k = k+1;
        if k<=length(I) % rescale
            v = D(:,I(k));
            if strcmp(normalization, 'clamp')
                v = clamp( 3*v/vmax, -1,1 );
                v = (v+1)/2;
            else
                v = rescale(v);
            end
            v = reshape(v,w1,w1,s);
            selx = (i-1)*(w1+pad)+1:(i-1)*(w1+pad)+w1;
            sely = (j-1)*(w1+pad)+1:(j-1)*(w1+pad)+w1;
            if s==1
                v = repmat(v,[1 1 3]);
            end
            H(selx,sely,:) = v;
        end
    end
end

H(end-pad+1:end,:,:) = [];
H(:,end-pad+1:end,:) = [];

if nargout==0
    clf;
    imagesc(H); axis image; axis off;
end

function y = clamp(x,a,b)

% clamp - clamp a value
%
%   y = clamp(x,a,b);
%
% Default is [a,b]=[0,1].
%
%   Copyright (c) 2004 Gabriel Peyré

if nargin<2
    a = 0;
end
if nargin<3
    b = 1;
end

if iscell(x)
    for i=1:length(x)
        y{i} = clamp(x{i},a,b);
    end
    return;
end

y = max(x,a);
y = min(y,b);
function v = getoptions(options, name, v, mendatory)

% getoptions - retrieve options parameter
%
%   v = getoptions(options, 'entry', v0, mendatory);
% is equivalent to the code:
%   if isfield(options, 'entry')
%       v = options.entry;
%   else
%       v = v0;
%   end
%
%   Copyright (c) 2007 Gabriel Peyre

if nargin<3
    error('Not enough arguments.');
end
if nargin<4
    mendatory = 0;
end

if isfield(options, name)
    v = eval(['options.' name ';']);
elseif mendatory
    error(['You have to provide options.' name '.']);
end 
function y = rescale(x,a,b)

% rescale - rescale data in [a,b]
%
%   y = rescale(x,a,b);
%
%   Copyright (c) 2004 Gabriel Peyr?

if nargin<2
    a = 0;
end
if nargin<3
    b = 1;
end

if iscell(x)
    for i=1:length(x)
        y{i} = rescale(x{i},a,b);
    end
    return;
end

m = min(x(:));
M = max(x(:));

if M-m<eps
    y = x;
else
    y = (b-a) * (x-m)/(M-m) + a;
end
