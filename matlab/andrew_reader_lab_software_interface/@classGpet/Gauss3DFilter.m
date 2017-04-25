% *********************************************************************
% Reconstruction Framework for Siemens Biograph mMR.  
% class: Gpet
% Authors: Martin Belzunce, Abolfazl Mehranian. Kings College London.
% Date: 08/02/2016
% *********************************************************************

function gf3d = Gauss3DFilter (objGpet, data, fwhm)
%  fwhm: convolution kernel size in cm

    if fwhm==0
        gf3d = data;
        return
    end
    if size(data,3)>1
        vox3dsz = objGpet.image_size.voxelSize_mm;
    else
        vox3dsz = objGpet.image_size.voxelSize_mm(1:2);
    end

    gsigmm=fwhm/sqrt(2^3*log(2));
    matsz=ceil(2*fwhm./vox3dsz);
    for i=1:size(matsz,2)
        if isequal(mod(matsz(i),2),0), matsz(i)=matsz(i)+1; end
    end
    padSize = (matsz-1)/2;
    bound=padSize.*vox3dsz;
    if size(data,3)>1
        [x,y,z] = meshgrid(-bound(2):vox3dsz(2):bound(2), -bound(1):vox3dsz(1):bound(1), -bound(3):vox3dsz(3):bound(3));
        h = exp(-(x.*x + y.*y + z.*z)/(2*gsigmm*gsigmm));
    else
        [x,y] = meshgrid(-bound(2):vox3dsz(2):bound(2), -bound(1):vox3dsz(1):bound(1));
        h = exp(-(x.*x + y.*y)/(2*gsigmm*gsigmm));
    end
    h = h/sum(h(:));
    numDims = length(padSize);
    idx = cell(numDims,1);
    for k = 1:numDims
        M = size(data,k);
        onesVector = ones(1,padSize(k));
        idx{k} = [onesVector 1:M M*onesVector];
    end
    b = data(idx{:});

    gf3d = convn(b,h, 'valid');
end