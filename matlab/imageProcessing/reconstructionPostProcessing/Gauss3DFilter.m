function gf3d = Gauss3DFilter (data, image_size, fwhm)
%  fwhm: convolution kernel size in cm

if fwhm==0
    gf3d = data;
    return
end
vox3dsz = image_size.voxelSize_mm;

gsigmm=fwhm/sqrt(2^3*log(2));
matsz=ceil(2*fwhm./vox3dsz);
for i=1:size(matsz,2)
    if isequal(mod(matsz(i),2),0), matsz(i)=matsz(i)+1; end
end
padSize = (matsz-1)/2;
bound=padSize.*vox3dsz;
[x,y,z] = meshgrid(-bound(2):vox3dsz(2):bound(2), -bound(1):vox3dsz(1):bound(1), -bound(3):vox3dsz(3):bound(3));
h = exp(-(x.*x + y.*y + z.*z)/(2*gsigmm*gsigmm));
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
