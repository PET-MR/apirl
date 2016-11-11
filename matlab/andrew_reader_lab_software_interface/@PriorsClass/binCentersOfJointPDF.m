function [binCenter,nBins,binwidth,binBoundery,fB] = binCentersOfJointPDF(ObjPrior,f,nBins,maxbin,d)

if nargin ==4
    d = 0;
else
    d = 1;
end
if isempty(maxbin)
    maxy = max(abs(f(:)))*1.2;
else
    maxy = maxbin;
end
%  maxy = maxbin;
miny = min(abs(f(:)))+1;


nBins = min(nBins,floor(maxy));
binBoundery = floor(linspace(miny,maxy,nBins+1));

binwidth = binBoundery(2) - binBoundery(1);

binCenter = floor(binBoundery(1:end-1) + binwidth/2);

if d
    fB = 0*f;
    
    for ic = 1:length(binBoundery)-1
        
        I = binBoundery(ic)<= f &  f < binBoundery(ic+1);
        fB(I) = binCenter(ic);
        
    end
else
    fB = [];
end
end