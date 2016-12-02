function [u,S] = SENSE_TV_ADMM(ObjMRI,arg)

if nargin==1, arg.save = 0; end

% default values
opt.PenaltyParameter = 0.1; %\rho
opt.RegualrizationParameter = 0.0005; % \lambda
opt.SigmaParameter = 0;% 0: TV, >0: NCX
opt.ADMM_niter = 50;
opt.SENSE_niter = 2;
opt.save = 0;
opt.display = 0;

opt = getFiledsFromUsersOpt(opt,arg);

% only for undersampled data
if ~ObjMRI.isUnderSampled
    fprintf('Only undersampled recon is supported\n')
    u = [];
    return
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

if save && ~ischar(m)
    N = min(floor(opt.ADMM_niter/n),opt.ADMM_niter);
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
    S =[];
else
    S =[];
end

[gamma_u,zu] = deal(zeros(prod(ObjMRI.CropedImageSize),ObjMRI.nS,'single'));

FHy = ObjMRI.FH(ObjMRI.kSpaceUnderSampled);
u = zeros(ObjMRI.nkSamples,'single');



if opt.display, figure ,end

for i = 1:opt.ADMM_niter
    
    RHS = FHy + ObjMRI.TransGraphGradUndoCrop(opt.PenaltyParameter*zu - gamma_u);
    u = ObjMRI.PCG(RHS, @(x) ObjMRI.FH(ObjMRI.F(x)) + ...
        opt.PenaltyParameter*ObjMRI.TransGraphGradUndoCrop(ObjMRI.GraphGradCrop(x)), u, opt.SENSE_niter, 1);
    
   if opt.SigmaParameter
       Lu = ObjMRI.RSS(zu);
       w = exp(-opt.SigmaParameter*Lu./ObjMRI.Magnitude(Lu+eps));
       w = repmat(w,[1,ObjMRI.nS]);
   else
       w = 1;
   end
    
    Du = ObjMRI.GraphGradCrop(u);
    z_tilde_u = Du + gamma_u/opt.PenaltyParameter;
    
    zu = ObjMRI.softThreshold(ObjMRI.RSS(z_tilde_u), z_tilde_u, opt.PenaltyParameter,opt.RegualrizationParameter, w);
    
    gamma_u = gamma_u + opt.PenaltyParameter*(Du - zu);
    
    
    if opt.display
        fprintf('Iteration: %d\n',i);
        
        if ObjMRI.is3D
           drawnow, imshow(abs(u(:,:,opt.display)),[]);
        else
            drawnow, imshow(abs(u));
        end
    end
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
    
end
end
