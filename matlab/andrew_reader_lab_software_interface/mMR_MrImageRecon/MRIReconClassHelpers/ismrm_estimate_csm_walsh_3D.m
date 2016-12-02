
function [csm] = ismrm_estimate_csm_walsh_3D(img, smoothing, chunks)
    %
    %   [csm] = ismrm_estimate_csm_walsh(img)
    %
    %   Estimates relative coil sensitivity maps from a set of coil images
    %   using the eigenvector method described by Walsh et al. (Magn Reson Med
    %   2000;43:682-90.)
    %
    %   INPUT:
    %     - img     [x,y,z,coil]   : Coil images
    %     - smooth  scalar       : Smoothing block size (defaults to 5)
    %     - chunks  scalar       : Number of chunks to split data in before
    %                              computing correlation/eigen decomposition
    %                              (Use if memory is an issue)
    %
    %   OUTPUT:
    %
    %     - csm     [x,y,z,coil]    : Relative coil sensitivity maps
    %
    %
    %   Code is based on an original implementation by Peter Kellman, NHLBI,
    %   NIH
    %
    %   Code made available for the ISMRM 2013 Sunrise Educational Course
    %
    %   Michael S. Hansen (michael.hansen@nih.gov)
    %
    %   Adapted for 3D by andrew.aitken@kcl.ac.uk
    %


    if nargin < 2,
        smoothing = 5;
    end

    if nargin < 3
        chunks = 1;
    end

    ncoils = size(img,4);

    % normalize by root sum of squares magnitude
    mag = sqrt(sum(img .* conj(img),4));
    s_raw=img ./ repmat(mag + eps,[1 1 1 ncoils]); clear mag;

    csm = zeros(size(img));

    num_chunks = ceil(size(img,3)/chunks);
    for chunk = 1:num_chunks
        fprintf('   Chunk %d of %d\n', chunk, num_chunks);
        start = 1 + (chunk-1)*chunks;
        finish = min((chunk)*chunks,size(img,3));

        % compute sample correlation estimates at each pixel location
        Rs=ismrm_correlation_matrix(s_raw(:,:,start:finish,:));


        % apply spatial smoothing to sample correlation estimates (NxN convolution)

        if smoothing
            d = size(Rs);

            [x,y,z] = ndgrid(-floor(d(1)/2):ceil(d(1)/2)-1, -floor(d(2)/2):ceil(d(2)/2)-1,-floor(d(3)/2):ceil(d(3)/2)-1);

            D0 = d/2*5/smoothing;

            Hx = exp(-x.^2/(2*D0(1).^2));
            Hy = exp(-y.^2/(2*D0(2).^2));
            Hz = exp(-z.^2/(2*D0(3).^2));

            H_lp = Hx.*Hy.*Hz;
            %             H_lp = repmat(H_lp, [1 1 1 d(4) d(5)]);

            for m=1:ncoils
                for n=1:ncoils
                    Rs(:,:,:,m,n) = ifftn(ifftshift(H_lp.*fftshift(fftn(Rs(:,:,:,m,n)))));
                end
            end
        end


        % compute dominant eigenvectors of sample correlation matrices
        csm(:,:,start:finish,:) =ismrm_eig_power(Rs); % using power method


    end
end


%Utility functions provided by Peter Kellman, NIH.
function [Rs]=ismrm_correlation_matrix(s)
    % function [Rs]=correlation_matrix(s);
    %
    % function correlation_matrix calculates the sample correlation matrix (Rs) for
    % each pixel of a multi-coil image s(y,x,coil)
    %
    % input:
    %    s   complex multi-coil image s(y,x,coil)
    % output:
    %    Rs  complex sample correlation matrices, Rs(y,x,coil,coil)

    %     ***************************************
    %     *  Peter Kellman  (kellman@nih.gov)   *
    %     *  Laboratory for Cardiac Energetics  *
    %     *  NIH NHLBI                          *
    %     ***************************************

    [rows,cols,nslices,ncoils]=size(s);
    Rs=zeros(rows,cols,nslices,ncoils,ncoils); % initialize sample correlation matrix to zero
    for i=1:ncoils
        for j=1:i-1
            Rs(:,:,:,i,j)=s(:,:,:,i).*conj(s(:,:,:,j));
            Rs(:,:,:,j,i)=conj(Rs(:,:,:,i,j)); % using conjugate symmetry of Rs
        end
        Rs(:,:,:,i,i)=s(:,:,:,i).*conj(s(:,:,:,i));
    end

end

function [v,d]=ismrm_eig_power(R)
    % function [v,d]=eig_power(R);
    %
    % vectorized method for calculating the dominant eigenvector based on
    % power method. Input, R, is an image of sample correlation matrices
    % where: R(y,x,:,:) are sample correlation matrices (ncoil x ncoil) for each pixel
    %
    % v is the dominant eigenvector
    % d is the dominant (maximum) eigenvalue

    %     ***************************************
    %     *  Peter Kellman  (kellman@nih.gov)   *
    %     *  Laboratory for Cardiac Energetics  *
    %     *  NIH NHLBI                          *
    %     ***************************************

    rows=size(R,1);cols=size(R,2);slices=size(R,3);ncoils=size(R,4);
    N_iterations=2;
    v=ones(rows,cols,slices,ncoils); % initialize e.v.

    d=zeros(rows,cols,slices);
    for i=1:N_iterations
        v=squeeze(sum(R.*repmat(v,[1 1 1 1, ncoils]),4));
        d=ismrm_rss(v);
        d( d <= eps) = eps;
        
        v=v./repmat(d,[1 1 1, ncoils]);
    end

    p1=angle(conj(v(:,:,:,1)));
    % (optionally) normalize output to coil 1 phase
    v=v.*repmat(exp(sqrt(-1)*p1),[1 1 1, ncoils]);
    v=conj(v);

end

function y = ismrm_rss(x,dim)
%
%   [mag] = ismrm_rss(samples, dim)
%
%   Computes root-sum-of-squares along a single dimension.
%
%
%   INPUT:
%     - x   : multi-dimensional array of samples
%     - dim : dimension of reduction; defaults to last dimension
%
%   OUTPUT:
%
%     - y       : root sum of squares result
%
%
%   Code made available for the ISMRM 2013 Sunrise Educational Course
% 
%   Michael S. Hansen (michael.hansen@nih.gov)
%   Philip J. Beatty (philip.beatty@sri.utoronto.ca)
%


if nargin==1
    dim=ndims(x);
else
    if isempty(dim); dim=ndims(x); end
end

y = squeeze(sqrt(sum(real(x).^2 + imag(x).^2,dim)));
end