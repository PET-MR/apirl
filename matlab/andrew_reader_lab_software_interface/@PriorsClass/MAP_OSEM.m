
function img = MAP_OSEM(ObjPrior,PET,Prompts,RS, SensImg,opt,Img )
if nargin==6
    Img = PET.ones;
end
% %
if ~isfield(opt,'start')
    opt.start = 0;
end

if opt.lambda==0
    fprintf('OP-MLEM\n');
else
    fprintf('MAP: %s\n',opt.prior)
    if strcmpi(opt.prior,'tikhonov') || strcmpi(opt.prior,'tv')
        fprintf('weight_method: %s\n',opt.weight_method)
        if strcmpi(opt.weight_method,'nl_joint_similarity')
           fprintf('n_modalities: %d\n',opt.n_modalities) 
        end
    end
end

if ~isfield(opt,'message')
	if opt.lambda==0
	opt.message = 'OP-MLEM';
	else
    opt.message = [opt.prior '-\lambda:' num2str(opt.lambda)];
	end
end

if opt.save
    if PET.image_size.matrixSize(3)==1
        IMG = zeros([PET.image_size.matrixSize(1:2),opt.nIter],'single');
    else
        if ~exist(opt.save_i,'file')
            mkdir(opt.save_i);
        end
		report(opt);
    end
end
% %
if opt.display
    figure
end
converged = 0;
relative_error = zeros(opt.nIter,1);
Img_old = single(Img);
i = 1;
while (i <= opt.nIter) && ~converged
    for j = 1:PET.nSubsets
        
        if opt.lambda
             dP = opt.lambda*ObjPrior.dPrior(Img,opt);
        else
            dP = 0;
        end
        
         if PET.image_size.matrixSize(3)==1
            Sen = SensImg(:,:,j)+dP+1e-5;
        else
            Sen = SensImg(:,:,:,j)+dP+1e-5;
        end       
        
        if PET.nSubsets==1
            Img = Img.*PET.PT(Prompts./(PET.P(Img)+ RS + 1e-5))./Sen;
        else
            Img = Img.*PET.PT(Prompts./(PET.P(Img,j)+ RS + 1e-5),j)./Sen;
        end

        Img = max(0,Img);
        if opt.display
            if PET.image_size.matrixSize(3)==1
                drawnow,subplot(121),imshow(Img,[]),colorbar
                if opt.lambda
                drawnow,subplot(122),imshow(dP,[]),colorbar
                end
            else
                drawnow,subplot(121),imshow(Img(:,:,50),[]),colorbar
                if opt.lambda
                drawnow,subplot(122),imshow(dP(:,:,50),[]),colorbar
                end
            end
            title(opt.message)
        end
    end
    fprintf('Iteration: %d\n',i);
    if opt.save
        if PET.image_size.matrixSize(3)==1
            IMG(:,:,i) = Img;
        else
            if ~mod(i,3)
                saveAt(Img,[opt.save_i '\Act_'  num2str(i + opt.start) '.dat']);%
            end
        end
    end
    i = i+1;
    [relative_error(i),converged] = converge_check(Img,Img_old,opt.tolerance);
    Img_old = Img;
end

img = opt;
if opt.save
    if PET.image_size.matrixSize(3)==1
        img.u_i = IMG;
    else
        saveAt(Img,[opt.save_i '\Act_'  num2str(i) '.dat']);
    end
end
if isfield(opt,'W0'),
    opt = rmfield(opt,'W0');
end


img.u = Img;
img.lambda = opt.lambda;
img.sWindowSize = ObjPrior.sWindowSize;
img.lWindowSize  =ObjPrior.lWindowSize;
img.psf = PET.PSF.Width;
img.relative_error = relative_error;

% remove large-sized fields
if isfield(img,'nl_weights')
    img = rmfield(img,'nl_weights');
end
if isfield(img,'je_weights')
    img = rmfield(img,'je_weights');
end
if isfield(img,'imgA')
    img = rmfield(img,'imgA');
end
if isfield(img,'normVectors')
    img = rmfield(img,'normVectors');
end

function saveAt(image, fname)
fid = fopen(fname,'w');
fwrite(fid,image,'float');
fclose(fid);


function  [relative_error,converged] = converge_check(x,xp,tolerance)
converged = 0;

relative_error= norm(x(:) - xp(:))/norm(xp(:));
if tolerance >= relative_error,
    converged =1;
end

function report(opt)

vfields = fieldnames(opt);
fid = fopen([opt.save_i '\parameters.txt'],'w');
for i = 1:length(vfields)
    s = opt.(vfields{i});
    if ~isa(s,'char')
        if numel(s)>1
            continue;
        end
        s = num2str(s);
    end
    fprintf(fid, '%s: %s\n',vfields{i},s);

end

fclose(fid);



