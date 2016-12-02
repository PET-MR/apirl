function [x,S] = PCG(ObjMRI,a,A,x0,Nit,P,opt)

% a: RHS of the normal equation, i.e. FH(mri data)
% A: FHF() handel object
% x0: initial image estimate
% Nit: number of sense iterations
% P: Precondictioner
% opt: options

% opt.save: 0 (default, no save), {n,m} saves every n iterates in
% output S, if image is 3D, m is the coronal slice number to be
% saved, if m = 0, all slices will be saved, {n,'dir'} saves
% every n iterates in the directory 'dir'

% opt.display: 0 (default, no display), m the coronal slice number
% for 3D images
if nargin<=6,
    opt.save = 0;
    opt.display = 0;
end
if isfield(opt,'save')
    if iscell(opt.save)
        save = 1;
        n = opt.save{1};
        m = opt.save{2};
    else
        save = 0;
    end
else
    save = 0;
end

if ~isfield(opt,'display'), opt.display = 0; end

if save && ~ischar(m)
    N = min(floor(Nit/n),Nit);
    if ObjMRI.is3D
        if m==0 % all silces are stored in S
            S = zeros([ObjMRI.nkSamples,N]);
        else % mth slice is stored in S
            S = zeros([ObjMRI.nkSamples(1:2),N]);
        end
    else
        S = zeros([ObjMRI.nkSamples(1:2),N]);
    end
elseif save && ischar(m)
    if ~exist(m,'dir'), mkdir(m); end
    S =[];
else
    S =[];
end
if opt.display, figure; end

a = P.*a;
u = x0./P;
r = a - P.*A(P.*u);
p = r;
for i=1:Nit
    q = P.*A(P.*p);
    alpha = r(:)'*r(:)/(p(:)'*q(:));
    u = u + alpha*p;
    rnew = r - alpha*q;
    p = rnew + rnew(:)'*rnew(:)/(r(:)'*r(:))*p;
    r = rnew;
    x = P.*u;
    
    % save --------------------
    if save
        if ~mod(i,n)
            if ~ischar(m)
                if ObjMRI.is3D
                    if m == 0
                        S(:,:,:,i) = u;
                    else
                        S(:,:,i) = u(:,:,m);
                    end
                else
                    S(:,:,i) = u;
                end
            else
                ObjMRI.saveAt(abs(u),[m '\' num2str(i)])
            end
        end
    end
%display-----------------------
if opt.display
    if ObjMRI.is3D
        drawnow, imshow(abs(u(:,:,opt.display)),[])
    else
        drawnow, imshow(abs(u),[])
    end
end
end
